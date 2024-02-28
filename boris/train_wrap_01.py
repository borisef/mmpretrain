import copy
import os.path
import os.path as osp
import sys

import mmcv
import numpy as np
import mmengine
from mmengine.config import Config

import cv2
import mmcv
from mmcv.transforms import Compose

import mmpretrain
import tools.train as train


config_file = "configs/config_resnet50_boris_try.py" # works
#config_file = "configs/config_resnext50_boris.py" # works
#config_file = "configs/config_vit_base32_boris.py" # works
#config_file = "configs/config_levit_256_boris.py" # works badly
#config_file = "configs/config_effnetv2_m_boris.py" # works weird
#config_file = "configs/config_beit_boris.py"# works badly




sys.argv.append(config_file)

train.main()

from mmengine.config import Config

cfg = Config.fromfile(config_file)
wd = cfg['work_dir']
ckpt = os.path.join(wd, "epoch_20.pth")



#inference
from mmpretrain import ImageClassificationInferencer
image = '/home/borisef/data/classification/caltech-101/small/test/car_side/image_0117.jpg'
model = mmpretrain.get_model(config_file, ckpt)
inferencer = ImageClassificationInferencer(model=config_file, pretrained=ckpt, device='cuda')
# Note that the inferencer output is a list of result even if the input is a single sample.
result = inferencer(image)[0]
print(result['pred_class'])
# You can also use is for multiple images.
image_list = [image] * 16
results = inferencer(image_list, batch_size=8)
print(len(results))
print(results[1]['pred_class'])