import imantics
import scipy as sp
import cv2
import os
from tqdm import tqdm
import json
from joblib import Parallel, delayed

image_id = 1
annotation_id = 1
coco_dataset = {'images': [], 'categories': [], 'annotations': []}
basedir = '/home/spadmin/data/lostandfound_processing/testset'
categories = {1: imantics.Category('void'), 2: imantics.Category('object')}
i = 0

# loads ego-vehicle mask
with open('/home/spadmin/Downloads/0000_02_Hanns_Klemm_Str_44_000000_000020_leftImg8bit_rgb (1).json', 'r') as f:
    coco_vehicle = json.load(f)
ego_mask = coco_vehicle['annotations'][0]['segmentation'][0]
ego_points = np.array(ego_mask).reshape((-1, 2)).round().astype('int')
vehicle_mask = np.zeros((1024, 2048, 1))
cv2.fillPoly(vehicle_mask, [ego_points], color=(100, 100, 100))
vehicle_mask = vehicle_mask[..., 0]

def process_image(file, image_id):
    """ Assumes that prediction and image file have similar prefix, see below"""
    prediction = cv2.imread(os.path.join(basedir, file), cv2.IMREAD_ANYDEPTH)
    # reduce noise (makes it easier to edit predictions)
    prediction = (prediction == 19).astype('int32')
    for _ in range(2):
        prediction = sp.ndimage.median_filter(prediction, size=12)
    # remove part where there is obvious street and ego-vehicle
    prediction[700:, 500:1750] = 0
    prediction[vehicle_mask != 0] = 1

    annotation = imantics.Image(width=2048, height=1024)
    for c in categories:
        annotation.add(imantics.Mask(prediction == c), category=categories[c])
    coco_dataset['images'].append({
        'id': image_id,
        'dataset_id': 1,
        'path': "/datasets/lostandfound/{}".format(file[:-10] + 'rgb.png'),
        'width': 2048,
        'height': 1024,
        'file_name': file[:-10] + 'rgb.png'})
    coco_annotations = annotation.coco()['annotations']
    for coco_annotation in coco_annotations:
        coco_annotation['id'] = 0
        coco_annotation['image_id'] = image_id
        coco_annotation['dataset_id'] = 1
    return coco_annotations

filelist = sorted([file for file in os.listdir(basedir) if file.endswith('pred.png')])

results = Parallel(n_jobs=8, require='sharedmem')(delayed(process_image)(file, i)
                                                  for i, file in enumerate(tqdm(filelist)))
coco_dataset['categories'] = [
    {
      "supercategory": None,
      "color": "#fd380c",
      "id": 1,
      "name": "void",
      "metadata": {}
    },
    {
      "supercategory": None,
      "color": "#e968dd",
      "id": 2,
      "name": "object",
      "metadata": {}
    }
  ]
# results is list of lists of annotations
for annotation_list in results:
    for annotation in annotation_list:
        annotation['id'] = annotation_id
        coco_dataset['annotations'].append(annotation)
        annotation_id += 1



with open(os.path.join(basedir, '../coco_pred.json'), 'w') as f:
    json.dump(coco_dataset, f, indent=2)
