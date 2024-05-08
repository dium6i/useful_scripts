"""
Author: Wei Qin, ChatGPT
Date: 2024-05-08
Description:
    Concatenating several images vertically into one image and fill the
    blank spaces with white pixels.
Update Log:
    2024-05-08: - File created.

"""

import os

import cv2
import numpy as np


def pad_image(img, target_width):
    """
    Fills the right side with white pixels to reach the target width, and
    returns the padded image.

    Args:
        img (numpy.ndarray): Image array.
        target_width (int): The desired width of the image after padding.

    Returns:
        padded (numpy.ndarray): The padded image as a NumPy array.
    """

    # Check if image is valid and width needs padding
    if img is None or img.shape[1] >= target_width:
        return img

    im_h, im_w, im_c = img.shape
    padding_width = target_width - im_w
    padding_array = np.full((im_h, padding_width, im_c), 255, dtype=np.uint8)
    padded = np.concatenate((img, padding_array), axis=1)

    return padded


def merge_images_in_groups(image_dir, group_size):
    """
    Group images according to group size and merge them vertically.

    Args:
        image_dir (str): Image directory.
        group_size (int): Number of images in group.

    Returns:
        None.
    """
    image_list = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]
    image_list.sort()  # Sort the files for consistent order

    num_groups = len(image_list) // group_size
    remainder = len(image_list) % group_size

    output_dir = os.path.join(os.path.dirname(image_dir), 'merged_images')
    os.makedirs(output_dir, exist_ok=True)

    imgs_info = os.path.join(os.path.dirname(image_dir), 'imgs_info.txt')
    with open(imgs_info, 'w') as f:  # image_file_name h w
        for i in range(num_groups):
            group_imgs = []  # Store image arrays in this group
            max_w = 0  # Width of merged_img

            for j in range(group_size):  # Iterate over images in this group
                file_name = image_list[i * group_size + j]
                img_path = os.path.join(image_dir, file_name)
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                max_w = max(max_w, w)
                f.write(f'{file_name} {h} {w} {i + 1}\n')
                group_imgs.append(img)

            group_imgs = [pad_image(im, max_w) for im in group_imgs]
            merged_img = cv2.vconcat(group_imgs)

            save_path = os.path.join(output_dir, f'{i + 1}.jpg')
            cv2.imwrite(save_path, merged_img)

        if remainder > 0:  # Merge remaining images (if any)
            remaining_imgs = []  # Store remaining image arrays
            max_w = 0  # Width of merged_img

            for k in range(remainder):  # Iterate over remaining images
                file_name = image_list[num_groups * group_size + k]
                img_path = os.path.join(image_dir, file_name)
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                max_w = max(max_w, w)
                f.write(f'{file_name} {h} {w} {num_groups + 1}\n')
                remaining_imgs.append(img)

            remaining_imgs = [pad_image(im, max_w) for im in remaining_imgs]
            merged_remaining_image = cv2.vconcat(remaining_imgs)

            save_path = os.path.join(output_dir, f'{num_groups + 1}.jpg')
            cv2.imwrite(save_path, merged_remaining_image)


if __name__ == '__main__':
    image_dir = 'dir/of/images'
    group_size = 6
    merge_images_in_groups(image_dir, group_size)
