import cv2

from dataloader.loader_base import LOADER_BASE


def CRAWLER(root, images, labels):
    pass


class LOADER(LOADER_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        label = self.labels[i]

        if self.show:
            cv2.imshow('sample', image)

        if self.make:
            pass
