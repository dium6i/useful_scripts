"""
Author: Wei Qin
Date: 2024-06-21
Description:
    Batch resizing images.
Update Log:
    2024-06-21: - File created.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import cv2
from tqdm import tqdm


def resize_image(imgs, save, short_side, img):
    """
    Single image resizing process.
    Args:
        imgs (string): Directory of source imasges.
        save (string): Directory to save resized image.
        img (string): Current image filename.

    Returns:
        None
    """
    img_path = os.path.join(imgs, img)
    im = cv2.imread(img_path)
    h, w, _ = im.shape

    if min(h, w) > short_side:
        if h > w:  # portrait
            dim = (short_side, int(short_side * h / w))
        else:  # landscape
            dim = (int(short_side * w / h), short_side)

        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

    base, ext = os.path.splitext(img)
    img = base + '.jpg' if ext == '.jfif' else img
    save_path = os.path.join(save, img)

    cv2.imwrite(save_path, im)


if __name__ == '__main__':
    imgs = 'path/of/source/images'
    save = 'path/to/save'
    short_side = 1500

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(resize_image, imgs, save, short_side, i)
            for i in os.listdir(imgs) if os.path.isfile(os.path.join(imgs, i))
        ]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Resizing...'):
            pass

    print('Image processing completed.')
