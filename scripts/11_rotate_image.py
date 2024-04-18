"""
Author: Wei Qin
Date: 2024-04-18
Description:
    Rotate the image with any angle while maintaining its integrity, and
    specify color to fill the missing background.
Update Log:
    2024-04-18: - File created.

"""

import os
import cv2
import numpy as np


def rotate_image(img, angle, color):
    # Get image dimensions
    img_h, img_w = img.shape[:2]

    # Calculate the rotation matrix
    r_matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)

    # Calculate the new dimensions of the rotated image
    cos_theta = np.abs(r_matrix[0, 0])
    sin_theta = np.abs(r_matrix[0, 1])
    new_w = int((img_h * sin_theta) + (img_w * cos_theta))
    new_h = int((img_h * cos_theta) + (img_w * sin_theta))

    # Adjust the rotation matrix for translation
    r_matrix[0, 2] += (new_w / 2) - (img_w / 2)
    r_matrix[1, 2] += (new_h / 2) - (img_h / 2)

    # Perform the rotation with adjusted dimensions
    rotated_img = cv2.warpAffine(
        img,
        r_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color)

    return rotated_img


if __name__ == '__main__':
    # Read image
    img_path = 'path/of/image'
    img = cv2.imread(img_path)

    # Define the rotation angle and the color to fill background
    angle = -28.6  # in degrees
    color = (0, 0, 0)  # black

    # Rotate the image
    rotated_img = rotate_image(img, angle, color)

    # Save rotated images
    basename, ext = os.path.splitext(img_path)
    rotated_img_path = basename + '_2' + ext
    cv2.imwrite(rotated_img_path, rotated_img)
