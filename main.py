import argparse
import os.path
from collections import deque
from glob import glob

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('--show', action='store_true', help='whether show')
parser.add_argument('--dirs', type=str, help='path to load image data')
parser.add_argument('--reid', type=str, choices=['on', 'off'], default='on', help='toggle DeepSORT ReID')
args = parser.parse_args()

if args.obb:
    args.model += '-obb'

deepsort = None
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
track_history = {}


def init_tracker(use_reid=True):
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE,
                        n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True,
                        use_reid=use_reid)


def compute_color_for_labels(label):
    if label == 0:
        color = (85, 45, 255)
    elif label == 2:
        color = (222, 82, 175)
    elif label == 3:
        color = (0, 204, 255)
    elif label == 5:
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_tracks(img, outputs, names):
    if outputs is None or len(outputs) == 0:
        return

    identities = outputs[:, -2].astype(int)
    object_ids = outputs[:, -1].astype(int)
    for key in list(track_history):
        if key not in identities:
            track_history.pop(key)

    for i, box in enumerate(outputs[:, :4].astype(int)):
        x1, y1, x2, y2 = box
        track_id = identities[i]
        cls_id = object_ids[i]
        color = compute_color_for_labels(cls_id)
        label = f"{track_id}:{names[cls_id]}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=64)
        track_history[track_id].appendleft(center)
        for j in range(1, len(track_history[track_id])):
            if track_history[track_id][j - 1] is None or track_history[track_id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
            cv2.line(img, track_history[track_id][j - 1], track_history[track_id][j], color, thickness)


model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

if args.show:
    if not args.obb:
        use_reid = args.reid == 'on'
        init_tracker(use_reid=use_reid)
        print(f"[Tracker] DeepSORT mode: {'ReID ON' if use_reid else 'ReID OFF (IOU-only)'}")
    with open(os.path.join('cfg', 'datasets', args.project + '.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    if args.dirs:
        img_dir = args.dirs
    else:
        img_dir = os.path.join(cfg['path'], cfg['test'])
    cls2name = cfg['names']
    results = model.predict(source=img_dir,
                            stream=True,
                            verbose=False)
    paused = False
    for r in tqdm(results, total=len(glob(os.path.join(img_dir, '*'))), ncols=80):
        if r.obb is not None:
            obb = r.obb
            for cls, box in zip(obb.cls, obb.xyxyxyxy.cpu()):
                cv2.polylines(r.orig_img, [np.asarray(box, dtype=int)], True, (0, 255, 0), 2)
                x1, y1 = box[0].int()
                # cv2.putText(r.orig_img, cls2name[cls.item()], (x1.item(), y1.item() - 5), 0, 1, (0, 255, 0), 2, 16)
        else:
            hbb = r.boxes
            # for cls, box in zip(hbb.cls, hbb.xyxy):
            #     x1, y1, x2, y2 = map(int, box)
            #     cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     # cv2.putText(r.orig_img, cls2name[cls.item()], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)

            if len(hbb.xywh):
                outputs = deepsort.update(hbb.xywh.cpu(), hbb.conf.cpu(), hbb.cls.cpu(), r.orig_img)
                draw_tracks(r.orig_img, outputs, cls2name)
        cv2.imshow('sample', r.orig_img)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == 27:
            break
        elif key == 32:  # Space key
            paused = not paused
