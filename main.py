import argparse
import os.path
from glob import glob

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8s')
parser.add_argument('-o', '--obb', action='store_true', help='whether obb')
parser.add_argument('-p', '--project', type=str, help='which object trained', default=None)
parser.add_argument('--show', action='store_true', help='whether show')
parser.add_argument('--auto', action='store_true', help='whether auto labeling')
parser.add_argument('--work', type=int, help='num of workers for multiprocessing', default=16)
parser.add_argument('--dirs', type=str, help='path to load image data')
args = parser.parse_args()

if args.obb:
    args.model += '-obb'


model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

if args.show:
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
            for cls, box in zip(hbb.cls, hbb.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(r.orig_img, cls2name[cls.item()], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)
        cv2.imshow('sample', r.orig_img)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == 27:
            break
        elif key == 32:  # Space key
            paused = not paused
