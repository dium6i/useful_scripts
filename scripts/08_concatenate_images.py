'''
Author: Wei Qin
Date: 2024-02-07
Description:
    Concatenate images horizontally with the same name in multiple 
    folders to easily compare the differences.
Update Log:
    2024-02-07: - File created.

'''

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


def draw_name(img, folder_name):
    '''
    Draw folder name on image.

    Args:
        img (numpy.ndarray): Read by OpenCV.
        folder_name (str): Folder name.

    Returns:
        img (numpy.ndarray): Image added text.
    '''
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = min(h, w) / 800
    org = (0, int(25 * fontScale))  # Bottom left coordinates of text.
    thickness_t = int(2 * fontScale)  # Top text.
    thickness_b = int(7 * fontScale)  # Bottom text.

    img = cv2.putText(img, folder_name, org, font, fontScale,
                      (255, 255, 255), thickness_b)
    img = cv2.putText(img, folder_name, org, font,
                      fontScale, (74, 64, 255), thickness_t)

    return img


def process_image(img_name, folders, data_dir, save_dir):
    '''
    Single image concatenation process.

    Args:
        img_name (str): Image name.
        folders (list): Folders containing other images with same name.
        data_dir (str): Data folder.
        save_dir (str): Save folder.

    Returns:
        None.
    '''
    images_to_concat = []

    # Find images with the same name in other folders
    for folder_name in folders:
        folder_path = os.path.join(data_dir, folder_name)
        img_path = os.path.join(folder_path, img_name)

        # Check if the image exists
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = draw_name(img, folder_name)

            # Add a separator ribbon if this is not the first image
            if not images_to_concat:
                height = img.shape[0]
                white_band = np.full(
                    (height, 15, 3), 255, dtype=np.uint8)  # Create a white ribbon
                images_to_concat.append(white_band)

            images_to_concat.append(img)

        else:
            print(f'File {img_path} not found.')
            return

    # Concatenate images if this image exists in all folders
    if len(images_to_concat) == len(folders) * 2 - 1:
        concat_img = np.hstack(images_to_concat)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, concat_img)


if __name__ == '__main__':
    data_dir = 'path/of/data/directory'
    folders = [
        'no_aug',
        'aug_degree',
        'aug_no_degree'
    ]

    save_dir = os.path.join(data_dir, 'compare')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Get all images list in the first folder
    ref_folder = os.path.join(data_dir, folders[0])
    images = [
        img for img in os.listdir(ref_folder)
        if img.endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Using ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_image, img_name, folders, data_dir, save_dir
            ) for img_name in images
        ]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing Images"):
            future.result()  # Get the result or exception of a function
