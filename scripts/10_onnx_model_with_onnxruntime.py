"""
Author: Wei Qin
Date: 2024-03-26
Description:
    Use the onnx model with the onnxruntime library for model inference.
Update Log:
    2024-03-26: - File created.
    2024-03-27: - Added the option to include category names in the results and
                  set inference threads when using CPU.
    2024-03-29: - Use cv2.dnn.NMSBoxes() to replace the customized NMS method.
    2024-04-30: - Keep the method of doing NMS in xyxy.
    2024-05-10: - Removed the option to include category names in the results.
                - Adapted inference for YOLOv8 classify model.
                - Modified image preprocessing to proportional scaling and padding.
                - Fixed issue with GPU inference failures.
    2024-06-07: - Added support for YOLOv10 (detect).
    2024-07-02: - Adjusted code structure.
                - Adapted inference for YOLOv8 pose model.
    2024-07-03: - Adjusted code structure.
    2024-07-04: - Adjusted code structure.
    2024-07-09: - Change bbox format from xywh to xyxy.
    2024-07-10: - Bug fixes.
    2024-07-11: - Bug fixes.

"""

import ast
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm


class YOLOv8:
    """
    This class handles the loading of the model, preprocessing of input images, postprocessing of model outputs,
    and visualization of the results. It supports different types of YOLOv8 models, including classification,
    object detection, and pose estimation.

    Attributes:
        model_path (str): The path to the ONNX model file.
        thread_num (int): The number of threads to use for inference. Default is -1 (use all available threads).
        conf_thres (float): The confidence threshold for object detection. Default is 0.5.
        iou_thres (float): The IoU threshold for non-max suppression. Default is 0.6.
        session (onnxruntime.InferenceSession): The loaded ONNX model.
        input_name (str): The name of the input node for the model.
        visualize (bool): Whether to visualize the results. Default is False.
        font_path (str): The path to the font file for visualization. Required if visualize is True.
    """

    def __init__(self, model_path, thread_num=-1, conf_thres=0.5, iou_thres=0.6):
        """
        Initialize the YOLOv8 class with the specified parameters.

        Args:
            model_path (str): Path to the ONNX model file.
            thread_num (int): Number of threads to use for inference. Default is -1.
            conf_thres (float): Confidence threshold for object detection. Default is 0.5.
            iou_thres (float): IoU threshold for non-max suppression. Default is 0.6.
        """
        self.thread_num = thread_num
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.initialize_model(model_path)

    def initialize_model(self, model_path):
        """
        Load and initialize the ONNX model.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        if self.thread_num == -1:
            self.thread_num = os.cpu_count()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.thread_num
        self.session = ort.InferenceSession(model_path, session_options)
        self.input_name = self.session.get_inputs()[0].name

        model_info = self.session.get_modelmeta().custom_metadata_map
        self.labels = ast.literal_eval(model_info['names'])  # {0: 'cat', ...}
        self.imgsz = ast.literal_eval(model_info['imgsz'])  # [640, 640]
        self.task = model_info['task']  # detect, classify, pose
        self.kpt_shape = (
            ast.literal_eval(model_info['kpt_shape'])
            if 'kpt_shape' in model_info
            else None
        )

    def preprocess(self, im):
        """
        Resize and pad the input image for model inference.

        Args:
            im (numpy.ndarray): Image array.

        Returns:
            im (numpy.ndarray): Preprocessed image data.
        """
        im = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        imgsz = self.imgsz[0]
        ratio = min(imgsz / h, imgsz / w)

        new_h, new_w = int(h * ratio), int(w * ratio)
        im = cv2.resize(im, (new_w, new_h))

        dh = imgsz - new_h  # Total added pixels in height.
        dw = imgsz - new_w  # Total added pixels in width.
        t, b = dh // 2, dh - (dh // 2)  # Added pixels in top and bottom.
        l, r = dw // 2, dw - (dw // 2)  # Added pixels in left and right.
        self.scale = (ratio, t, l)  # Scale info used to restore bbox.

        color = [127, 127, 127]  # Pad with gray pixels
        im = cv2.copyMakeBorder(im, t, b, l, r, cv2.BORDER_CONSTANT, value=color)

        im = np.array(im) / 255.0
        im = np.transpose(im, (2, 0, 1))  # Channel first
        im = np.expand_dims(im, axis=0).astype(np.float32)

        return im

    def classify_postproc(self, outs):
        """
        Postprocess of classify model.

        Args:
            outs (list): Inference results from the model.

        Returns:
            results (list): Filtered and formatted results. [id, name, score]
        """
        class_id = np.argmax(outs)
        class_name = self.labels[class_id]
        score = outs[class_id]

        return [[class_id, class_name, score]]

    def detect_postproc(self, outs):
        """
        Postprocess of detect model.

        Args:
            outs (list): Inference results from the model.

        Returns:
            results (list): Filtered and formatted results.
                            [[id, name, score, [(xmin, ymin), (w, h)]], ...].
        """
        outs = np.transpose(outs)

        max_scores = np.max(outs[:, 4:], axis=1)
        class_ids = np.argmax(outs[:, 4:], axis=1)
        valid_indices = np.where(max_scores >= self.conf_thres)
        outs = np.hstack((
            outs[:, :4][valid_indices],
            max_scores[valid_indices].reshape(-1, 1),
            class_ids[valid_indices].reshape(-1, 1),
        ))  # [[x, y, w, h, score, class_id], ...]

        # Restore bboxes to original size
        x, y = outs[:, 0].copy(), outs[:, 1].copy()
        w, h = outs[:, 2].copy(), outs[:, 3].copy()
        s, t, l = self.scale
        outs[:, 0] = ((x - w / 2) - l) / s
        outs[:, 1] = ((y - h / 2) - t) / s
        outs[:, 2] = w / s
        outs[:, 3] = h / s
        outs[:, :4] = outs[:, :4].astype(int)

        boxes = outs[:, :4]
        scores = outs[:, -2]
        class_ids = outs[:, -1]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        results = []
        for i in indices:
            class_id = int(class_ids[i])
            class_name = self.labels[int(class_ids[i])]
            score = round(float(scores[i]), 4)
            box = [(int(boxes[i][0]), int(boxes[i][1])), 
                   (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3]))]

            results.append([class_id, class_name, score, box])

        return results

    def pose_postproc(self, outs):
        """
        Postprocess of pose model.

        Args:
            outs (list): Inference results from the model.

        Returns:
            results (list): Filtered and formatted results.
                            [[id, name, score, [(xmin, ymin), (w, h)], [(x1, y1), (x2, y2), ...]], ...].
        """
        nc = len(self.labels)  # number of classes
        outs = np.transpose(outs)

        max_scores = np.max(outs[:, 4:4 + nc], axis=1)
        class_ids = np.argmax(outs[:, 4:4 + nc], axis=1)
        valid_indices = np.where(max_scores >= self.conf_thres)
        outs = np.hstack((
            outs[:, :4][valid_indices],
            max_scores[valid_indices].reshape(-1, 1),
            class_ids[valid_indices].reshape(-1, 1),
            outs[:, 4 + nc:][valid_indices]
        ))  # [[x, y, w, h, score, class_id, kpts], ...]

        # Restore bboxes and keypoints to original size
        s, t, l = self.scale

        x, y = outs[:, 0].copy(), outs[:, 1].copy()
        w, h = outs[:, 2].copy(), outs[:, 3].copy()
        outs[:, 0] = ((x - w / 2) - l) / s
        outs[:, 1] = ((y - h / 2) - t) / s
        outs[:, 2] = w / s
        outs[:, 3] = h / s

        x_cols = np.arange(6, outs.shape[1], 2)
        y_cols = np.arange(7, outs.shape[1], 2)
        kx, ky = outs[:, x_cols].copy(), outs[:, y_cols].copy()
        outs[:, x_cols] = (kx - l) / s
        outs[:, y_cols] = (ky - t) / s

        outs[:, :4] = outs[:, :4].astype(int)
        outs[:, 6:] = outs[:, 6:].astype(int)

        boxes = outs[:, :4]
        scores = outs[:, 4]
        class_ids = outs[:, 5]
        kpts = outs[:, 6:]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        results = []
        for i in indices:
            class_id = int(class_ids[i])
            class_name = self.labels[int(class_ids[i])]
            score = round(float(scores[i]), 4)
            box = [(int(boxes[i][0]), int(boxes[i][1])), 
                   (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3]))]
            kpt = [(int(kpts[i][j]), int(kpts[i][j + 1])) 
                   for j in range(0, len(kpts[i]), 2)]

            results.append([class_id, class_name, score, box, kpt])

        return results

    def postprocess(self, outs):
        """
        Postprocess the model inference results.

        Args:
            outs (list): Raw inference results from the model.

        Returns:
            results (list): Filtered and formatted results.
                            [[id, name, score, [(xmin, ymin), (w, h)]], ...].
        """
        outs = np.squeeze(outs)

        if self.task == 'classify':
            results = self.classify_postproc(outs)

        elif self.task == 'detect':
            results = self.detect_postproc(outs)

        elif self.task == 'pose':
            results = self.pose_postproc(outs)

        else:
            print('Unsupported task.')
            results = None

        return results

    def model_infer(self, im):
        """
        Model inference process.

        Args:
            im (numpy.ndarray): Image array.

        Returns:
            results (list): Inference results after postprocessing.
            dt (float): Inference time.
        """
        t0 = time.time()
        image_data = self.preprocess(im)
        outputs = self.session.run(None, {self.input_name: image_data})
        results = self.postprocess(outputs)
        dt = time.time() - t0

        if self.im_count == 1:
            print(f'{self.task.capitalize()} Results:')
            list(map(print, results))
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
        # Convert OpenCV image array to PIL image object
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        # Determines the line width of bboxes
        im_w, im_h = image.size
        lw = int(max(im_w, im_h) * 0.003) + 1 # line width

        colorset = [
            (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
            (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
            (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
            ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
            (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
            (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
            (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
            (217, 134, 177), (133, 111, 102), 
        ]

        for i in results:
            label_id, label_name, conf, ((xmin, ymin), (xmax, ymax)) = i[:4]
            kpts = i[4:]  # empty list if it is not a pose model
            color = colorset[label_id % len(colorset)]

            # Draw bbox
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=lw)

            # Draw label
            text = f'{label_name} {conf * 100:.2f}%'
            if min(im_w, im_h) < 600:
                th = 12
            else:
                th = int(max(im_w, im_h) * 0.015)

            font = ImageFont.truetype(self.font_path, th)
            tw = draw.textlength(text, font=font)

            x1 = xmin
            y1 = ymin - th - lw
            x2 = xmin + tw + lw
            y2 = ymin

            if y1 < 10:  # Label-top out of image
                y1 = ymin
                y2 = ymin + th + lw

            if x2 > im_w:  # Label-right out of image
                x1 = im_w - tw - lw
                x2 = im_w

            draw.rectangle(
                [(x1, y1), (x2 + lw, y2)], 
                fill=color, 
                width=lw)
            draw.text(
                (x1 + lw, y1 + lw), 
                text, 
                font=font, 
                fill=(255, 255, 255))

            # Draw keypoints
            if kpts:
                for idx, j in enumerate(kpts[0]):
                    kpt_color = colorset[idx % len(colorset)]
                    # top-left and bottom-right, lw as radius
                    tl = (j[0] - lw, j[1] - lw)
                    bw = (j[0] + lw, j[1] + lw) 
                    draw.ellipse([tl, bw], fill=kpt_color)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
        self.visualize = visualize if self.task != 'classify' else False
        self.font_path = font_path

        # numpy array as input
        if isinstance(im, np.ndarray):
            self.im_count = 1
            self.img_path = False
            results = self.model_infer(im)
            self.save_visualization(im, results)

        # path as input
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

        # unsupported input
        else:
            print('Please double check your input.')
            results = None

        return results


class YOLOv10(YOLOv8):
    """
    This class handles the loading of the model, preprocessing of input images, postprocessing of model outputs,
    and visualization of the results. It supports different types of YOLOv10 models, including classification,
    object detection, and pose estimation.

    Attributes:
        model_path (str): The path to the ONNX model file.
        thread_num (int): The number of threads to use for inference. Default is -1 (use all available threads).
        conf_thres (float): The confidence threshold for object detection. Default is 0.5.
        iou_thres (float): The IoU threshold for non-max suppression. Default is 0.6.
        session (onnxruntime.InferenceSession): The loaded ONNX model.
        input_name (str): The name of the input node for the model.
        visualize (bool): Whether to visualize the results. Default is False.
        font_path (str): The path to the font file for visualization. Required if visualize is True.
    """

    def detect_postproc(self, outs):
        """
        Postprocess of detect model.

        Args:
            outs (list): Inference results from the model.

        Returns:
            results (list): Filtered and formatted results.
                            [[id, name, score, [(xmin, ymin), (w, h)]], ...].
        """
        scores = outs[:, 4]
        indices = np.where(scores > self.conf_thres)[0]
        outs = outs[indices, :]

        # From xyxy to xywh
        x1, y1 = outs[:, 0].copy(), outs[:, 1].copy()
        x2, y2 = outs[:, 2].copy(), outs[:, 3].copy()
        w, h = x2 - x1, y2 - y1

        # Restore bboxes to original size
        s, t, l = self.scale
        outs[:, 0] = (x1 - l) / s
        outs[:, 1] = (y1 - t) / s
        outs[:, 2] = w / s
        outs[:, 3] = h / s

        boxes = outs[:, :4]
        scores = outs[:, -2]
        class_ids = outs[:, -1]

        results = []
        for i in indices:
            class_id = int(class_ids[i])
            class_name = self.labels[int(class_ids[i])]
            score = round(float(scores[i]), 4)
            box = [(int(boxes[i][0]), int(boxes[i][1])), 
                   (int(boxes[i][2]), int(boxes[i][3]))]

            results.append([class_id, class_name, score, box])

        return results


if __name__ == '__main__':
    image_dir = 'path/of/image/directory'  # Image file or directory
    model_path = 'path/of/model'  # Path of model
    font_path = 'path/of/font'  # Path of ttf font, required if visualize
    save_dir = 'path/to/save'  # Path to save visualized results

    model = YOLOv8(model_path)
    # model = YOLOv10(model_path)
    results = model.predict(
        image_dir,
        save_dir,
        visualize=True,
        font_path=font_path)
