import copy
import os
import sys

from mmengine.config import Config

import mmpretrain
import tools.visualization.browse_dataset as browse_dataset


#------INPUTS----------
config_file = "configs/config_resnet50_boris.py" # works
cfg = Config.fromfile(config_file)
wd = cfg['work_dir']
WORK_DIR = "WORK_DIR" #/home/borisef/Runs/mmpretrain/small/out_test" #The directory to save the file containing evaluation metrics.
if(WORK_DIR == "WORK_DIR"):
    WORK_DIR = os.path.join(wd,"out_test")

PHASE = "train"
OUTPUT_DIR = os.path.join(WORK_DIR,"viz_pipeline_" + PHASE ) # The directory to save the result visualization images
SHOW_NUMBER = 10 # None==all
MODE = 'pipeline' #['original', 'transformed', 'concat', 'pipeline']
SHOW_INTERRVAL = 0.1# time delay
RESCALE_FACTOR = 0.5
CHANNEL_ORDER = "BGR"
#------VIZ----------
sys.argv.append(config_file)

if(OUTPUT_DIR is not None): sys.argv.append("--output-dir");sys.argv.append(OUTPUT_DIR)

sys.argv.append("--phase");sys.argv.append(PHASE)
sys.argv.append("--mode");sys.argv.append(MODE)
sys.argv.append("--rescale-factor");sys.argv.append(str(RESCALE_FACTOR))
sys.argv.append("--channel-order");sys.argv.append(CHANNEL_ORDER)


if(SHOW_NUMBER is not None):
    # sys.argv.append("--show")
    sys.argv.append("--show-number");sys.argv.append(str(int(SHOW_NUMBER)))

sys.argv.append("--show-interval"); sys.argv.append(str(float(SHOW_INTERRVAL)))


browse_dataset.main()
