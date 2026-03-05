import json
import os
from glob import glob

import cv2
import numpy as np

from dataloader.loader_base import LOADER_BASE


def CRAWLER(root, images, labels):
    images.extend(sorted(glob(os.path.join(root, 'ds', 'img', '*.jpg'))))
    labels.extend(sorted(glob(os.path.join(root, 'ds', 'ann', '*.json'))))


class LOADER(LOADER_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        with open(self.labels[i], 'r', encoding='utf-8') as file:
            line = json.load(file)
        tags = line['tags']
        size = line['size']
        objects = line['objects']

        labels = []
        for obj in objects:
            cls = obj['classTitle']
            x1, y1 = obj['points']['exterior'][0]
            x2, y2 = obj['points']['exterior'][1]
            label = [x1, y1, x2, y2, self.name2cls[cls]]
            labels.append(label)

        if self.show:
            for label in labels:
                x1, y1, x2, y2, cls = label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 3)
                cv2.putText(image, self.cls2name[cls], (x1, y1 - 5), 0, 1, (0, 255, 0), 2, 16)
            cv2.imshow('sample', image)

        if self.make:
            labels = np.asarray(labels, dtype=np.float32)
            if len(labels):
                labels[..., 2] -= labels[..., 0]
                labels[..., 3] -= labels[..., 1]
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
            self.installer(i, image, labels)

        return i, i
