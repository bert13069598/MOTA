import json
import os
from glob import glob

import cv2
import numpy as np

from dataloader.loader_base import LOADER_BASE


def CRAWLER(root, images, labels):
    for tvt in ['train', 'val', 'test']:
        tvt_path = os.path.join(root, tvt)
        images.extend(sorted(glob(os.path.join(tvt_path, 'img', '*.png'))))
        labels.extend(sorted(glob(os.path.join(tvt_path, 'ann', '*.json'))))


class LOADER(LOADER_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        with open(self.labels[i], 'r', encoding='utf-8') as file:
            line = json.load(file)
        objects = line['objects']

        classes = []
        boxes = []
        for obj in objects:
            cls = obj['classTitle']
            box = obj['points']['exterior']
            classes.append(self.name2cls[cls])
            boxes.append(box)

        if self.show:
            for cls, box in zip(classes, boxes):
                cv2.polylines(image, [np.asarray(box, dtype=int)], True, (0, 255, 0), 2)
                cv2.putText(image, self.cls2name[cls], (box[0][0], box[0][1] - 5), 0, 1, (0, 255, 0), 2, 16)
            cv2.imshow('sample', image)

        if self.make:
            if len(boxes):
                valid = np.array([len(box) == 4 for box in boxes], dtype=bool)
                boxes = [boxes[i] for i, v in enumerate(valid) if v]
                boxes = np.asarray(boxes, dtype=np.float32)
                classes = np.asarray(classes, dtype=np.float32)[valid]

                labels = np.hstack((classes.reshape(-1, 1), boxes.reshape(-1, 8)))
            else:
                labels = np.zeros((0, 9), dtype=np.float32)
            self.installer(i, image, labels)

        return i, i
