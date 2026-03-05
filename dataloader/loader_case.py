import importlib
import os

import yaml

from dataloader import loader


def data_select(args):
    images, labels = [], []
    if args.data not in loader.loader_map:
        raise ValueError('No matching dataset')

    data = loader.data[args.data]
    with open(os.path.join('cfg', 'datasets', data + '.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    root_path = cfg['path']
    cls2name = cfg['names']
    save_path = args.path if args.path else root_path

    which = importlib.import_module(f'dataloader.loader.{loader.loader_map[args.data]}')
    which.CRAWLER(root_path, images, labels)
    datasets = which.LOADER(args, images, labels, save_path, cls2name)
    return save_path, datasets
