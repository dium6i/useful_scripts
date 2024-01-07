'''
Author: Wei Qin
Date: 2023-12-15
Description:
    Crop images(VOC dataset) according to annotations
    to check whether the label is wrong.
Update Log:
    2023-12-15: File created.
    2024-01-07: Use multi-threading to speed up.

'''

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm


def process_image(params, img):
    '''
    Single image cropping process.

    Args:
        params (dict): Parameters.
        img (string): Image file name like "sample.jpg".

    Returns:
        None.
    '''
    snippet = img.split('.')
    base = '.'.join(snippet[:-1])
    ext = snippet[-1]

    img_path = os.path.join(params['imgs_dir'], img)
    xml_path = os.path.join(params['xmls_dir'], base + '.xml')

    im = cv2.imread(img_path)
    i = 1  # Starting index of cropped images.

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        label = obj.find('name').text

        if not params['crop_all'] and label not in params['cropped_labels']:
            continue

        obj_dir = os.path.join(params['save_dir'], label)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)

        bndbox = obj.find('bndbox')
        box = [
            int(float(bndbox.find('xmin').text)),
            int(float(bndbox.find('ymin').text)),
            int(float(bndbox.find('xmax').text)),
            int(float(bndbox.find('ymax').text))
        ]
        croped = im[box[1]:box[3], box[0]:box[2]]

        cv2.imwrite(os.path.join(obj_dir, f'{base}_{i}.{ext}'), croped)
        i += 1


def run(params):
    '''
    Main process.

    Args:
        params (dict): Parameters.

    Returns:
        None.
    '''
    t = time.strftime('%Y%m%d_%H%M%S')
    params['imgs_dir'] = os.path.join(params['data'], params['imgs'])
    params['xmls_dir'] = os.path.join(params['data'], params['xmls'])
    params['save_dir'] = os.path.join(params['data'], params['save'], t)
    imgs_list = os.listdir(params['imgs_dir'])

    # Using ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, params, img)
                   for img in imgs_list]
        for _ in tqdm(
                as_completed(futures),
                total=len(imgs_list),
                desc='Cropping'):
            pass

    print('Cropping completed.')


if __name__ == '__main__':
    # Setting parameters
    params = {
        'data': 'path/of/dataset',  # Contains imgs and xmls
        'imgs': 'JPEGImages',  # Folder of images
        'xmls': 'Annotations',  # Folder of labels
        'save': 'cropped',  # Folder to save cropped images
        'crop_all': False,  # Whether to crop all labels
        'cropped_labels': ['cat', 'dog']  # Labels need to be cropped
    }

    run(params)
