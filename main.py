import argparse
import os.path
# from concurrent.futures import ThreadPoolExecutor
from glob import glob
# from typing import Tuple

import cv2
import numpy as np
# import torch
import yaml
# from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

# from dataloader.loader_base import LOADER

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


# def collate_fn(batch):
#     paths, image = zip(*batch)
#     image = np.asarray(image)
#     image = (image[..., ::-1]).transpose(0, 3, 1, 2)
#     image = np.ascontiguousarray(image)
#     image = torch.from_numpy(image).float()
#     image /= 255
#     return paths, image


# def annotate_label(rescale_factor: Tuple[bool, float, float],
#                    path: str,
#                    r):
#     path = '.'.join(path.split('.')[:-1] + ['txt'])
#     w_h, scale, pad = rescale_factor
#     if r.obb is not None:
#         obb = r.obb
#         x = obb.xyxyxyxyn.cpu().numpy().reshape(-1, 8)
#         x[:, w_h::2] *= scale
#         x[:, w_h::2] -= pad
#
#         with open(path, 'w', encoding='utf-8') as f:
#             for cls, box in zip(obb.cls, x):
#                 x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten()
#                 f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
#                     int(cls.item()), x1, y1, x2, y2, x3, y3, x4, y4))
#     else:
#         hbb = r.boxes
#         x = hbb.xywhn.cpu().numpy()
#         x[:, 2 + w_h] *= scale
#         x[:, int(w_h)] = 0.5 + (x[:, int(w_h)] - 0.5) * scale
#
#         with open(path, 'w', encoding='utf-8') as f:
#             for cls, box in zip(hbb.cls, x):
#                 cx, cy, w, h = box
#                 f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(cls), cx, cy, w, h))


model = YOLO(f'./runs/{args.model}/{args.project}/weights/best.pt')

# if args.auto:
#     datasets = LOADER(args)
#     dataloader = DataLoader(datasets,
#                             batch_size=args.work,
#                             num_workers=min(30, args.work),
#                             collate_fn=collate_fn
#                             )
#
#     executor = ThreadPoolExecutor()
#     progress = tqdm(total=len(datasets), ncols=80)
#
#     w0, h0 = datasets.wh0
#     w_h = w0 > h0
#     r = max(w0, h0) / min(w0, h0)
#     pad = abs(w0 - h0) / 2 / min(w0, h0)
#     rescale_factor = w_h, r, pad
#
#     for paths, tensor in dataloader:
#         progress.update(len(tensor))
#         results = model.predict(tensor, verbose=False)
#         executor.map(lambda args: annotate_label(rescale_factor, *args), zip(paths, results))

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
                cv2.putText(r.orig_img, cls2name[cls.item()], (x1.item(), y1.item() - 5), 0, 1, (0, 255, 0), 2, 16)
        else:
            hbb = r.boxes
            for cls, box in zip(hbb.cls, hbb.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(r.orig_img, cls2name[cls.item()], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)
        cv2.imshow('sample', r.orig_img)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == 27:
            break
        elif key == 32:  # Space key
            paused = not paused
