"""
Author: Wei Qin
Date: 2024-06-03
Description:
    PPOCR model inference and results visualization.
    The current version is only suitable for model inference using FastDeploy.
Update Log:
    2024-06-03: - File created.
    2024-06-07: - Optimized code structure.

"""

import os

import cv2
import fastdeploy as fd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ocr_model(img, model_path):
    """
    Initialize OCR model.

    Args:
        img (numpy.ndarray): Image array.

    Returns:
        model_path (str): Model path.
    """
    det_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    det_option.use_openvino_backend()
    rec_option.use_openvino_backend()

    det_model = fd.vision.ocr.DBDetector(
        model_file=os.path.join(model_path, 'det/inference.pdmodel'),
        params_file=os.path.join(model_path, 'det/inference.pdiparams'),
        runtime_option=det_option)
    # det_model.postprocessor.det_db_unclip_ratio = 0.9

    rec_model = fd.vision.ocr.Recognizer(
        model_file=os.path.join(model_path, 'rec/inference.pdmodel'),
        params_file=os.path.join(model_path, 'rec/inference.pdiparams'),
        label_path=os.path.join(model_path, 'ppocr_keys_v1.txt'),
        runtime_option=rec_option)

    ocr_model = fd.vision.ocr.PPOCRv3(
        det_model=det_model,
        cls_model=None,
        rec_model=rec_model)

    results = ocr_model.predict(img)

    return results


def draw_bbox(img, idx, bbox, alpha=0.5):
    """
    Draw a bounding box with a transparent fill on an image.

    Args:
        img (numpy.ndarray): Array of original image.
        idx (int): Index of bbox.
        bbox (list): List of four corner points of the bounding box [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        alpha (float): The transparency factor of the fill. (0 is fully transparent)

    Returns:
        output (numpy.ndarray): Array of image with bbox drawn.
    """
    colorset = [ # RGB
        (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
        (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
        (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
        ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
        (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
        (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
        (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
        (217, 134, 177), (133, 111, 102), 
    ]
    color = colorset[idx % len(colorset)]
    color = color[::-1]  # BGR to RGB

    overlay = img.copy()
    output = img.copy()

    # Convert bbox to numpy array and reshape for cv2 functions
    points = np.array(bbox, np.int32).reshape((-1, 1, 2))

    # Fill the polygon
    cv2.fillPoly(overlay, [points], color)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Draw the outline of the bounding box
    cv2.polylines(output, [points], isClosed=True, color=color, thickness=1)

    return output, color


if __name__ == '__main__':
    img_path = 'path/of/image'
    img = cv2.imread(img_path)

    # Model inference
    model_path = 'path/of/models/directory'
    results = ocr_model(img, model_path)

    # Rearrange and filter results
    ocr_results = [[list(zip(results.boxes[i][0::2], results.boxes[i][1::2])),  # boxes
                    results.text[i],  # text
                    round(results.rec_scores[i], 4)]  # confidence
                   for i, _ in enumerate(results.boxes) if results.rec_scores[i] > 0.5]

    # Create a white image for visualizing the OCR results
    h, w, _ = img.shape
    result_im = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(result_im)
    font = ImageFont.truetype('path/of/font', size=0.0115 * h)

    # Visualize OCR results
    for i, res in enumerate(ocr_results):
        bbox, text, score = res
        img, color = draw_bbox(img, i, bbox)
        draw.text(bbox[-1], text, fill=color, font=font)

    result_img = np.array(result_im)
    vis_img = cv2.hconcat([img, result_img])

    # Save Visualization result
    dirname = os.path.dirname(img_path)
    base, ext = os.path.splitext(os.path.basename(img_path))
    cv2.imwrite(os.path.join(dirname, base + '_vis.jpg'), vis_img)
