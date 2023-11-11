import copy
import os.path
import mmpretrain
from mmengine.config import Config
from mmpretrain import ImageClassificationInferencer


config_file = "configs/config_resnet50_boris.py" # works


cfg = Config.fromfile(config_file)
wd = cfg['work_dir']
ckpt = os.path.join(wd, "epoch_20.pth")



#inference
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