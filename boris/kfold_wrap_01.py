import sys

import tools.atraf.kfold_cross_valid as kfold_cross_valid

num_splits = 2


config_file = "configs/config_resnet50_boris_kf_1.py"  # works




sys.argv.append(config_file)
sys.argv.append('--num-splits')
sys.argv.append(str(num_splits))
sys.argv.append('--wrap_dataset_func')
sys.argv.append('whole_dataset') #by_session_wrap_dataset, wrap_dataset, whole_dataset


kfold_cross_valid.main()
