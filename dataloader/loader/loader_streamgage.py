import json
import os
from glob import glob

import cv2
import numpy as np

from dataloader.loader_base import LOADER_BASE


def CRAWLER(root, images, labels):
    for tv in ['Training', 'Validation']:
        tvt_path = os.path.join(root, tv)
        images_path = sorted(glob(os.path.join(tvt_path, '01.원천데이터', '*')))
        images_path = [f for f in images_path if os.path.isdir(f)]
        labels_path = sorted(glob(os.path.join(tvt_path, '02.라벨링데이터', '*')))
        labels_path = [f for f in labels_path if os.path.isdir(f)]
        for image_path in images_path:
            images.extend(sorted(glob(os.path.join(image_path, '*'))))
        for label_path in labels_path:
            labels.extend(sorted(glob(os.path.join(label_path, '*'))))


class LOADER(LOADER_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        with open(self.labels[i], 'r', encoding='utf-8') as file:
            line = json.load(file)
        annos = line['annotations']

        labels = []
        for anno in annos:
            cls = anno['class_id']
            xywh = json.loads(anno['relative_coordinates'])
            cx = xywh['center_x']
            cy = xywh['center_y']
            w = xywh['width']
            h = xywh['height']
            label = [cx, cy, w, h, cls]
            labels.append(label)

        width = line['image']['width']
        height = line['image']['height']

        if self.show:
            for label in labels:
                cx, cy, w, h, cls = label
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                x1 *= width
                x2 *= width
                y1 *= height
                y2 *= height
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 3)
                cv2.putText(image, self.cls2name[cls], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)
            cv2.imshow('sample', image)

        if self.make:
            labels = np.asarray(labels, dtype=np.float32)
            if len(labels):
                labels[..., 0] -= labels[..., 2] / 2
                labels[..., 1] -= labels[..., 3] / 2
                labels[..., 0:4:2] *= width
                labels[..., 1:4:2] *= height
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
            self.installer(i, image, labels)

        return i, i
