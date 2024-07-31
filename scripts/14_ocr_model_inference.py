# -*- coding: utf-8 -*-
"""
Author: Wei Qin
Date: 2024-06-03
Description:
    PPOCR model inference and results visualization.
Update Log:
    2024-06-03: - File created.
    2024-06-07: - Optimized code structure.
    2024-06-26: - Optimized code structure.
    2024-07-10: - Changed the inference engine from FastDeploy to RapidOCR.
    2024-07-11: - Bug fixes.
    2024-07-12: - Bug fixes.
    2024-07-16: - Optimized output format.
    2024-07-31: - Optimized code structure.

"""

import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm


class OCR:
    def __init__(
            self,
            text_score=0.5,
            det_model_path=None,
            det_unclip_ratio=1.6,
            rec_model_path=None,
            intra_op_num_threads=-1):
        """
        RapidOCR has built-in general OCR models(PPOCRv4).
        If costomized models need to be used, det_model_path and rec_model_path
        should be specified.
        """
        self.colorset = [ # RGB
            (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
            (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
            (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
            ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
            (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
            (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
            (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
            (217, 134, 177), (133, 111, 102), 
        ]

        self.engine = RapidOCR(
            text_score=text_score,
            det_model_path=det_model_path,
            det_unclip_ratio=det_unclip_ratio,
            rec_model_path=rec_model_path,
            intra_op_num_threads=intra_op_num_threads)

    def model_infer(self, im):
        """
        Model inference and results postprocess.

        Args:
            im (numpy.ndarray): Image array.

        Returns:
            results (list): Inference results after postprocessing.
            dt (float): Inference time.
        """
        t0 = time.time()
        res, _ = self.engine(im, use_det=True, use_cls=False, use_rec=True)
        dt = time.time() - t0

        # Rearrange and filter res
        results = []
        if res:
            for result in res:
                bbox, text, score = result
                box_wh = (bbox[1][0] - bbox[0][0], bbox[-1][1] - bbox[0][1])
                results.append([bbox, box_wh, text, float(score)])

        if self.im_count == 1:
            print(f'OCR Results (first 10):')
            list(map(print, results[:10]))
            print(f'Inference time: {dt * 1000:.2f} ms')
            return results
        else:
            return results, dt

    def draw_boxes(self, im, results):
        """
        Visualize boxes based on filtered results.

        Args:
            im (numpy.ndarray): Image array.
            results (list): Filtered results.

        Returns:
            im (numpy.ndarray): Visualized image.
        """
        h, w, _ = im.shape
        # create a white image to display results
        white_im = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(white_im)

        # Visualize OCR results
        for i, res in enumerate(results):
            bbox, wh, text, score = res

            # Draw box on original image
            color = self.colorset[i % len(self.colorset)][::-1]
            overlay = im.copy()
            # convert bbox and reshape for cv2 functions
            points = np.array(bbox, np.int32).reshape((-1, 1, 2))
            # draw box on characters
            cv2.fillPoly(overlay, [points], color)
            # blend images for transparent box
            cv2.addWeighted(overlay, 0.5, im, 0.5, 0, im)
            # draw outline of box
            cv2.polylines(im, [points], isClosed=True, color=color, thickness=1)

            # Draw text on result image
            font = ImageFont.truetype(self.font_path, size=wh[1])
            draw.text(bbox[0], text, fill=color, font=font)

        white_im = np.array(white_im)
        vis_img = cv2.hconcat([im, white_im])

        return vis_img

    def save_visualization(self, im, results):
        """
        Create a save folder in the directory where the current image file is
        located, and save the visualized images. When input is numpy.ndarray,
        set the code file as an image file.

        Args:
            im (numpy.ndarray): Image array.
            results (list): Inference results after postprocessing.

        """
        if self.visualize:
            if not self.save_dir:
                # os.path.abspath(__file__) is the path of current code file.
                current_file = self.img_path if self.img_path else os.path.abspath(__file__)
                current_file_dir = os.path.dirname(current_file)
                self.save_dir = os.path.join(current_file_dir, 'visualized')
                os.makedirs(self.save_dir, exist_ok=True)

            save_name = os.path.basename(self.img_path) if self.img_path else 'visualized.jpg'
            save_path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(save_path, self.draw_boxes(im, results))
            if self.im_count == 1:
                print(f'Visualized result saved at: {save_path}')

    def predict(self, im, save_dir=None, visualize=False, font_path=None):
        """
        Run model inference on the input image.

        Args:
            im (np.ndarray|str): Image array or image(s) directory.

        Returns:
            results (list): Inference results after postprocessing.
        """
        self.save_dir = save_dir
        if self.save_dir and os.path.isfile(save_dir):
            self.save_dir = os.path.join(os.path.dirname(save_dir), 'visualized')
            os.makedirs(self.save_dir, exist_ok=True)
        self.visualize = visualize
        self.font_path = font_path

        # Numpy array as input
        if isinstance(im, np.ndarray):
            self.im_count = 1
            self.img_path = False
            results = self.model_infer(im)
            self.save_visualization(im, results)

        # Path as input
        elif isinstance(im, str):
            if os.path.isfile(im):  # Single image
                self.im_count = 1
                self.img_path = im
                im_array = cv2.imread(im)
                results = self.model_infer(im_array)
                self.save_visualization(im_array, results)

            else:  # Directory of images
                self.im_count = -1
                img_list = [i for i in os.listdir(im) 
                    if os.path.isfile(os.path.join(im, i))]
                t = 0
                for img in tqdm(img_list, total=len(img_list), desc='Processing'):
                    img_path = os.path.join(im, img)
                    self.img_path = img_path
                    im_array = cv2.imread(img_path)
                    results, dt = self.model_infer(im_array)
                    self.save_visualization(im_array, results)
                    t += dt

                print(f'Inference time:\n    Total: {t * 1000:.2f} ms')
                print(f'    Avg: {t * 1000 / len(img_list):.2f} ms')
                if self.visualize:
                    print(f'Visualized results saved at: {self.save_dir}')
                results = None

        # Unsupported input
        else:
            print('Please double check your input.')
            results = None

        return results


if __name__ == '__main__':
    model = OCR(det_unclip_ratio=2.0)
    model.predict(
        'path/of/image/directory',  # Image file or directory
        save_dir='path/to/save',  # Path to save visualized results
        visualize=True, 
        font_path='path/of/font',  # Path of ttf font, required if visualize
    )
