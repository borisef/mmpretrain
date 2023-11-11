import copy
import os.path
import os.path as osp
import sys

import mmcv
import numpy as np
import mmengine
from mmengine.config import Config

import mmpretrain
import tools.test as test
import tools.analysis_tools.confusion_matrix as confmat


#------INPUTS----------
config_file = "configs/config_resnet50_boris.py" # works
cfg = Config.fromfile(config_file)
wd = cfg['work_dir']
WORK_DIR = "WORK_DIR" #/home/borisef/Runs/mmpretrain/small/out_test" #The directory to save the file containing evaluation metrics.
if(WORK_DIR == "WORK_DIR"):
    WORK_DIR = os.path.join(wd,"out_test")


ckpt_file = os.path.join(wd, "epoch_20.pth")
OUT_ITEM = 'metrics' # To specify the content of the test results file, and it can be “pred” or “metrics”.
OUT  = os.path.join(WORK_DIR,"out.pkl") #pickle
SHOW_DIR = os.path.join(WORK_DIR,"show_dir") # The directory to save the result visualization images
TO_SHOW_SKIP_INTERVAL = 1 # None = do nothing
WAIT_TIME = 0.01 # if slow or fast show
CM_SHOW_DIR =  os.path.join(WORK_DIR,"cm")

#------TEST----------
#python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
sys.argv.append(config_file)
sys.argv.append(ckpt_file)
sys_argv_backup = sys.argv.copy()

if(OUT is not None): sys.argv.append("--out");sys.argv.append(OUT)
sys.argv.append("--out-item");sys.argv.append(OUT_ITEM)

if(WORK_DIR is not None):sys.argv.append("--work-dir");sys.argv.append(WORK_DIR)

if(TO_SHOW_SKIP_INTERVAL is not None):
    sys.argv.append("--show")
    sys.argv.append("--interval");sys.argv.append(str(int(TO_SHOW_SKIP_INTERVAL)))
    sys.argv.append("--wait-time"); sys.argv.append(str(WAIT_TIME))
if(SHOW_DIR is not None):
    sys.argv.append("--show-dir");sys.argv.append(SHOW_DIR)

test.main()

#------CONF_MAT----------
sys.argv = sys_argv_backup.copy()
sys.argv.append("--show")
sys.argv.append("--include-values")
sys.argv.append("--show-path"); sys.argv.append(CM_SHOW_DIR)
confmat.main()
