import os.path

from mmpretrain.engine.hooks.visualization_hook import VisualizationHook

import math
import os.path as osp
from typing import Optional, Sequence

from mmengine.fileio import join_path
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from mmpretrain.registry import HOOKS
from mmpretrain.structures import DataSample

from typing import Dict, Optional #BE
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import cv2


@HOOKS.register_module()
class VisualizationExtraHook(VisualizationHook):
    def __init__(self,
                 enable=False,
                 interval: int = 5000,
                 show: bool = False,
                 out_dir: Optional[str] = None,
                 conf_mat_params = None,
                 **kwargs):
        super(VisualizationExtraHook, self).__init__(enable,interval,show,out_dir,**kwargs)

        default_conf_mat_params =  {'save_csv': True, 'save_img': False}
        if(conf_mat_params is not None):
            default_conf_mat_params.update(conf_mat_params)
        self.conf_mat_params = default_conf_mat_params

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:

        if ('confusion_matrix/result' in metrics):
            M = metrics['confusion_matrix/result']
            cf_matrix = M.numpy()
            classes = runner.val_evaluator.dataset_meta['classes']
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                                 columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            # plt.savefig("/home/borisef/tmp/confusion_matrix.png")
            # from PIL import Image

            # Open the saved image
            # image = Image.open("confusion_matrix.png")

            # Convert the image to a numpy array
            # rgb_array = np.array(image)
            # Get the current figure as a numpy array
            fig = plt.gcf()
            fig.canvas.draw()

            # Convert the figure to a numpy array
            rgb_array = np.array(fig.canvas.renderer._renderer)

            runner.visualizer.add_image(
                name='confusion_matrix/result', image=rgb_array, step=runner.epoch)

            plt.close()

            if(self.conf_mat_params['save_csv'] or self.conf_mat_params['save_img'] ):
                wd = runner.cfg['work_dir']
                if(self.out_dir is not None ):
                    wd = self.out_dir
                if(not os.path.exists(os.path.join(wd,'confusion_matrix'))):
                    os.mkdir(os.path.join(wd,'confusion_matrix'))
                if(self.conf_mat_params['save_csv']):
                    # save cm in txt file in wd/confusion_matrix/epoch_....txt
                    df_cm.to_csv(os.path.join(wd,'confusion_matrix', str(runner.epoch) + '_cm.csv'))
                if (self.conf_mat_params['save_img']):
                    # save cm in txt file in wd/confusion_matrix/epoch_.....png
                    cv2.imwrite(os.path.join(wd,'confusion_matrix', str(runner.epoch) + '_cm.png'), rgb_array)

        pass
