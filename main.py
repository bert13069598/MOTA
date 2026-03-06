import argparse
import os.path
from collections import deque
from glob import glob

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
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
args = parser.parse_args()

if args.obb:
    args.model += '-obb'

deepsort = None
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
track_history = {}


class OBBTracker:
    def __init__(self, max_age=15, max_cost=0.45, w_pos=0.55, w_ang=0.25, w_size=0.20):
        self.max_age = max_age
        self.max_cost = max_cost
        self.w_pos = w_pos
        self.w_ang = w_ang
        self.w_size = w_size
        self.next_id = 1
        self.tracks = []

    @staticmethod
    def _angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    @staticmethod
    def _xywhr_from_polygon(poly):
        rect = cv2.minAreaRect(poly.astype(np.float32))
        (cx, cy), (w, h), deg = rect
        return np.array([cx, cy, max(w, 1.0), max(h, 1.0), np.deg2rad(deg)], dtype=np.float32)

    @staticmethod
    def _corners_from_xywhr(xywhr):
        cx, cy, w, h, a = xywhr
        rect = ((float(cx), float(cy)), (max(float(w), 1.0), max(float(h), 1.0)), float(np.rad2deg(a)))
        return cv2.boxPoints(rect)

    @staticmethod
    def _sanitize_xywhr(xywhr):
        arr = np.asarray(xywhr, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 5:
            pad = np.zeros(5, dtype=np.float32)
            pad[:arr.shape[0]] = arr
            arr = pad
        arr = arr[:5]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[2] = max(abs(float(arr[2])), 1.0)
        arr[3] = max(abs(float(arr[3])), 1.0)
        arr[4] = float(np.arctan2(np.sin(arr[4]), np.cos(arr[4])))
        return arr.astype(np.float32)

    def _predict(self, track):
        pred = track["state"].copy()
        pred[:4] = pred[:4] + track["vel"][:4]
        pred[4] = pred[4] + track["vel"][4]
        return pred

    def _cost(self, pred, det, diag):
        eps = 1e-6
        pred = self._sanitize_xywhr(pred)
        det = self._sanitize_xywhr(det)
        pos_cost = np.linalg.norm(pred[:2] - det[:2]) / max(diag, 1.0)
        ang_cost = abs(self._angle_diff(pred[4], det[4])) / np.pi
        size_cost = 0.5 * (
            abs(np.log((det[2] + eps) / (pred[2] + eps))) +
            abs(np.log((det[3] + eps) / (pred[3] + eps)))
        )
        total = self.w_pos * pos_cost + self.w_ang * ang_cost + self.w_size * size_cost
        if not np.isfinite(total):
            return 1e6
        return float(total)

    def _init_track(self, det):
        track = {
            "id": self.next_id,
            "state": self._sanitize_xywhr(det["xywhr"]),
            "vel": np.zeros(5, dtype=np.float32),
            "cls": det["cls"],
            "conf": det["conf"],
            "poly": det["poly"].copy(),
            "miss": 0,
            "hits": 1,
        }
        self.next_id += 1
        self.tracks.append(track)

    def update(self, detections, image_shape):
        img_h, img_w = image_shape
        diag = float(np.hypot(img_w, img_h))

        for t in self.tracks:
            t["pred"] = self._predict(t)

        if not self.tracks:
            for det in detections:
                self._init_track(det)
            return [t for t in self.tracks if t["miss"] == 0]

        if not detections:
            for t in self.tracks:
                t["miss"] += 1
            self.tracks = [t for t in self.tracks if t["miss"] <= self.max_age]
            return []

        n_t, n_d = len(self.tracks), len(detections)
        cost = np.full((n_t, n_d), fill_value=1e6, dtype=np.float32)
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                if track["cls"] != det["cls"]:
                    continue
                cost[i, j] = self._cost(track["pred"], det["xywhr"], diag)

        cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
        row_idx, col_idx = linear_sum_assignment(cost)
        matched_t, matched_d = set(), set()
        for r_idx, c_idx in zip(row_idx, col_idx):
            if cost[r_idx, c_idx] > self.max_cost:
                continue
            track = self.tracks[r_idx]
            det = detections[c_idx]
            prev = track["state"].copy()
            det_state = self._sanitize_xywhr(det["xywhr"])
            delta = det_state - prev
            delta[4] = self._angle_diff(det_state[4], prev[4])
            track["vel"] = 0.7 * track["vel"] + 0.3 * delta
            track["state"] = det_state
            track["cls"] = det["cls"]
            track["conf"] = det["conf"]
            track["poly"] = det["poly"].copy()
            track["miss"] = 0
            track["hits"] += 1
            matched_t.add(r_idx)
            matched_d.add(c_idx)

        for i, track in enumerate(self.tracks):
            if i not in matched_t:
                track["miss"] += 1
                track["poly"] = self._corners_from_xywhr(track["pred"])
                track["state"] = track["pred"]

        for j, det in enumerate(detections):
            if j not in matched_d:
                self._init_track(det)

        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_age]
        return [t for t in self.tracks if t["miss"] == 0]


def init_tracker():
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
                        use_cuda=True)


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


def extract_obb_detections(obb):
    polys = obb.xyxyxyxy.cpu().numpy()
    clss = obb.cls.cpu().numpy().astype(int)
    confs = obb.conf.cpu().numpy()

    if hasattr(obb, "xywhr") and obb.xywhr is not None:
        xywhrs = obb.xywhr.cpu().numpy()
    else:
        xywhrs = None

    detections = []
    for i, poly in enumerate(polys):
        xywhr_raw = xywhrs[i] if xywhrs is not None else OBBTracker._xywhr_from_polygon(poly)
        xywhr = OBBTracker._sanitize_xywhr(xywhr_raw)
        detections.append({
            "xywhr": xywhr,
            "poly": np.asarray(poly, dtype=np.float32),
            "cls": int(clss[i]),
            "conf": float(confs[i]),
        })
    return detections


def draw_obb_tracks(img, tracks, names):
    for t in tracks:
        color = compute_color_for_labels(t["cls"])
        poly = np.asarray(t["poly"], dtype=np.int32)
        cv2.polylines(img, [poly], True, color, 2)
        x, y = poly[0]
        label = f'{t["id"]}:{names[t["cls"]]}'
        cv2.putText(img, label, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

if args.show:
    if not args.obb:
        init_tracker()
    else:
        obb_tracker = OBBTracker()
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
            detections = extract_obb_detections(obb)
            tracked = obb_tracker.update(detections, r.orig_img.shape[:2])
            draw_obb_tracks(r.orig_img, tracked, cls2name)
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
