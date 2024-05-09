"""
Author: Wei Qin
Date: 2024-05-09
Description:
    Splitting vertically concatenated images into original images.
Update Log:
    2024-05-09: - File created.

"""

import os
from collections import defaultdict

import cv2


def read_images_info(info_file_path):
    """
    Read an images info file that records information such as file name, width,
    height, and group for each image.

    Args:
        info_file_path (str): Path of images info file.

    Returns:
        images_info (dict): Dict recording width and height of each image.
        group_info (dict): Dict recording which images are contained in each
                           group.
    """
    images_info = {}  # {'img_name': (h, w), ...}
    group_info = defaultdict(list)  # {'1': [img1.jpg, img2.jpg, ...], ...}

    with open(info_file_path, 'r') as f:
        for line in f:
            name, h, w, group = line.strip().split()
            images_info[name] = (int(h), int(w))
            group_info[group].append(name)

    return images_info, group_info


def split_images(imgs, images_info, group_info):
    """
    Split the merged image based on the info of each image.

    Args:
        imgs (str): Directory of merged images.
        images_info (dict): Dict recording width and height of each image.
        group_info (dict): Dict recording which images are contained in each
                                           group.
    Returns:
        None.
    """
    for img in os.listdir(imgs):
        group_key = os.path.splitext(img)[0]  # '1', '2', ...
        im = cv2.imread(os.path.join(imgs, img))

        h = 0  # Starting point for splitting current image.
        for i in group_info[group_key]:
            im_h, im_w = images_info[i]
            img_i = im[h:h + im_h, 0:im_w]
            cv2.imwrite(os.path.join(save, i), img_i)
            h += im_h


if __name__ == '__main__':
    imgs = 'path/of/merged/images'
    save = 'path/of/splited/images'
    os.makedirs(save, exist_ok=True)

    info_file_path = 'path/of/images/info/file'
    images_info, group_info = read_images_info(info_file_path)
    split_images(imgs, images_info, group_info)
