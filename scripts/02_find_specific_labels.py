'''
Author: Wei Qin
Date: 2023-12-15
Description:
    Find images with specific labels in VOC dataset.
Update Log:
    2023-12-15: File created.
    2024-01-06: Change code structure.

'''

import os
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm


def search_labels(params):
    params['imgs_dir'] = os.path.join(params['dataset'], params['imgs'])
    params['xmls_dir'] = os.path.join(params['dataset'], params['xmls'])

    i = 0
    found = []
    skip = False

    for img in tqdm(os.listdir(params['imgs_dir']), desc='Searching'):
        if i == params['num']:  # Found enough images.
            break

        base = '.'.join(img.split('.')[:-1])
        xml_path = os.path.join(params['xmls_dir'], base + '.xml')
        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.iter('object'):
            if skip:  # Current image has been counted.
                continue

            if obj.find('name').text in params['labels']:
                i += 1
                found.append(img)
                skip = True  # Skip current image.

        skip = False

    if params['num'] <= params['thresh']:
        print('\n'.join(found))

    return found


def copy_files(params, found):
    ph = ''
    if params['save_dir'] == params['dataset']:
        ph = 'has_target_labels'

    copied_imgs = os.path.join(params['save_dir'], ph, 'JPEGImages')
    copied_xmls = os.path.join(params['save_dir'], ph, 'Annotations')
    os.makedirs(copied_imgs, exist_ok=True)
    os.makedirs(copied_xmls, exist_ok=True)

    for img in found:
        base = '.'.join(img.split('.')[:-1])
        img_path = os.path.join(params['imgs_dir'], img)
        xml_path = os.path.join(params['xmls_dir'], base + '.xml')

        shutil.copyfile(img_path,
                        os.path.join(copied_imgs, img))
        shutil.copyfile(xml_path,
                        os.path.join(copied_xmls, base + '.xml'))


if __name__ == '__main__':
    # Setting parameters
    params = {
        'dataset': r'D:\01_hzwq\15_jlx_dnb\02_dnb',
        'imgs': 'JPEGImages',  # Folder of images
        'xmls': 'Annotations',  # Folder of labels
        'labels': ['dnb_blank'],  # Labels hope to find.
        'num': 15,  # Number of images hope to find.
        'thresh': 20,  # Print results if "num" is less than this.
        'copy': False,  # Whether to copy found images to other directory.
        'save_dir': r'D:\01_hzwq\15_jlx_dnb\02_dnb'
    }

    # Main process
    found = search_labels(params)
    if params['copy']:
        copy_files(params, found)
