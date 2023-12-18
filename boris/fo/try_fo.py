import os

import fiftyone as fo
import fiftyone.zoo as foz

foz.list_zoo_models(["detection"])

foz.list_zoo_datasets(["detection"])

#foz.load_zoo_dataset('coco-2017', split = 'validation', max_samples = 50)


#voc1 = foz.load_zoo_dataset('voc-2012', split = 'validation', max_samples = 50)
#voc1

qs = foz.load_zoo_dataset('quickstart')
qs

foz.list_zoo_models()

foz.list_zoo_models()
clip_model = foz.load_zoo_model('clip-vit-base32-torch')
clip_model
dataset = foz.load_zoo_dataset('quickstart')
dataset.evaluate_detections("predictions",eval_key="eval")
session = fo.launch_app(dataset, auto=False)
dataset.compute_metadata()
sample = dataset.first()
sample.filepath
sample.eval_fp
dataset[0]
dataset[ '/home/borisef/fiftyone/quickstart/data/000880.jpg']
sample.field_names
sample.ground_truth
dataset.list_aggregations()
dataset.bounds()
dataset.aggregate('bounds')
dataset.bounds('uniqness')
dataset.bounds("uniqueness")
dataset.bounds(["eval_tp"])
dataset.bounds(["eval_tp","eval_fp"])

#dataset.bounds?

dataset.distinct("ground_truth.detections.label")
dataset.distinct("ground_truth.detections")
dataset.distinct("ground_truth")
dataset.distinct("ground_truth.detections.label")[0]
dataset[10]
dataset[10:20]
v = dataset[10:20]
v
v.classes
v.dataset_name
v.list_view_stages()
v10 = dataset.skip(11).limit(12)
v10 = dataset.skip(11).limit(12).shuffle()
v10
type(v10)
type(dataset)
v3 = dataset
dataset
v3 = dataset.match(("ground_truth.detections.label")=='cat')
from fiftyone import ViewField as F
v3 = dataset.match(F("ground_truth.detections.label")=='cat')
type(v3)
v3 = dataset.match(F("ground_truth.detections.label")=='person')
v3 = dataset.match(F("ground_truth.detections.label").contains('person')
                   )
v3
F("ground_truth.detections.label")
v3 = dataset.match(F("ground_truth.detections.label").contains('person'))
dataset.save_view("person_view", v3)
dataset.list_saved_views()