import glob
import os

data = sorted(glob.glob(os.path.join('cfg/datasets', "*.yaml")), key=os.path.getctime)
data = [data.split('/')[-1].replace('.yaml', '') for data in data]

# file name
# dataloader/loader/loader_car.py -> loader_car
loader_map = {i: 'loader_' + d for i, d in enumerate(data)}
