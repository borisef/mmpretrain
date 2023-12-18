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



#config_file = "configs/config_mocov2_resnet50_8xb32-coslr-200e_in1k.py"# moco
config_file = "/home/borisef/projects/mm/mmpretrain/boris/configs/config_resnet50_8xb32-linear-steplr-100e_in1k.py"

sys.argv.append(config_file)

train.main()
