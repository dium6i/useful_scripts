'''
Author: Wei Qin
Date: 2023-12-15
Description:
    Crop images(VOC dataset) according to labels.
Update Log:
    2023-12-15: File created.

'''

import os
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm


data = 'path/of/voc/dataset'
imgs = os.path.join(data, 'JPEGImages')
xmls = os.path.join(data, 'Annotations')
crop = os.path.join(data, 'croped')
if not os.path.exists(crop):
    os.mkdir(crop)

crop_all = False  # Weather to crop all labels.
croped_labels = [  # Labels need to be croped.
    'cat',
    'dog',
    'people',
    'kite'
]

for img in tqdm(os.listdir(imgs), desc="Cropping"):
    snippet = img.split('.')
    base_name = '.'.join(snippet[:-1])
    ext = snippet[-1]

    img_path = os.path.join(imgs, img)
    xml_path = os.path.join(xmls, base_name + '.xml')

    tree = ET.parse(xml_path)
    root = tree.getroot()

    im = cv2.imread(img_path)
    i = 1 # Starting index of cropped images. 

    for obj in root.iter('object'):
        label = obj.find('name').text

        if not crop_all and label not in croped_labels:
            continue

        # Create folder based on label. 
        obj_dir = os.path.join(crop, label)
        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)

        # Crop labels. 
        bndbox = obj.find('bndbox')
        box = [
            int(float(bndbox.find('xmin').text)),
            int(float(bndbox.find('ymin').text)),
            int(float(bndbox.find('xmax').text)),
            int(float(bndbox.find('ymax').text))
        ]
        croped = im[box[1]:box[3], box[0]:box[2]]
        
        # Save cropped images. 
        cv2.imwrite(os.path.join(obj_dir, f'{base_name}_{i}.{ext}'), croped)
        i += 1
