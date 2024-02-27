'''
Author: Wei Qin
Date: 2024-01-06
Description:
    Find images with specific labels in VOC dataset.
Update Log:
    2024-01-06: - File created.
    2024-01-07: - Add function description.
    2024-02-23: - Use multithreading to speed up searching and
                  copying process.
                - Adjusted code structure.
    2024-02-27: - Corrected spelling errors.

'''

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm


def reprocess_params(params):
    '''
    Reprocess parameters.

    Args:
        params (dict): Parameters.

    Returns:
        params (dict): Parameters.
    '''
    ph = ''  # Placeholder
    if params['save_dir'] == params['dataset']:
        ph = 'has_target_labels'

    imgs_save = os.path.join(params['save_dir'], ph, 'JPEGImages')
    xmls_save = os.path.join(params['save_dir'], ph, 'Annotations')
    os.makedirs(imgs_save, exist_ok=True)
    os.makedirs(xmls_save, exist_ok=True)

    params['imgs_save'] = imgs_save
    params['xmls_save'] = xmls_save

    return params


def check_image(img, params):
    '''
    Check whether an image contains target label(s).

    Args:
        img (str): Image filename.
        params (dict): Parameters.

    Returns:
        img (str): Image filename.
    '''
    base = '.'.join(img.split('.')[:-1])  # Avoid filename like abc.def.jpg
    xml_path = os.path.join(params['xmls_dir'], base + '.xml')
    if not os.path.exists(xml_path):
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        if obj.find('name').text in params['labels']:
            return img

    return None


def search_labels(params):
    '''
    Search target label name in dataset.

    Args:
        params (dict): Parameters.

    Returns:
        found (list): List of image names containing target label names.
    '''
    params['imgs_dir'] = os.path.join(params['dataset'], params['imgs'])
    params['xmls_dir'] = os.path.join(params['dataset'], params['xmls'])

    found = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(check_image, img, params)
            for img in os.listdir(params['imgs_dir'])
        ]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Searching'):
            result = future.result()
            if result:
                found.append(result)
                if len(found) == params['num']:  # Found enough images.
                    break

    if params['num'] <= params['thresh'] and params['num'] != -1:
        print('\n'.join(found))

    return found


def copy_file(img, params):
    '''
    Copy process of a single image.

    Args:
        img (str): Image filename.
        params (dict): Parameters.

    Returns:
        None.
    '''
    base = '.'.join(img.split('.')[:-1])
    img_path = os.path.join(params['imgs_dir'], img)
    xml_path = os.path.join(params['xmls_dir'], base + '.xml')

    shutil.copyfile(img_path, os.path.join(params['imgs_save'], img))
    shutil.copyfile(xml_path, os.path.join(params['xmls_save'], base + '.xml'))


def copy_files(params, found):
    '''
    Main process of copying files.

    Args:
        params (dict): Parameters.
        found (list): List of image names containing target label names.

    Returns:
        None.
    '''
    ph = ''  # Placeholder
    if params['save_dir'] == params['dataset']:
        ph = 'has_target_labels'

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(copy_file, img, params) for img in found]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Copying'):
            pass


if __name__ == '__main__':
    # Setting parameters
    params = {
        'dataset': 'path/of/dataset', # Contains imgs and xmls
        'imgs': 'JPEGImages',  # Folder of images
        'xmls': 'Annotations',  # Folder of labels
        'labels': ['dnb_blank'],  # Labels hoping to find
        'num': 15,  # Number of images hoping to find (-1 for all)
        'thresh': 20,  # Print results if "num" is less than this
        'copy': False,  # Whether to copy found images to other directory
        'save_dir': 'path/to/save/copied/files'
    }
    params = reprocess_params(params)

    # Main process
    found = search_labels(params)
    if params['copy']:
        copy_files(params, found)
