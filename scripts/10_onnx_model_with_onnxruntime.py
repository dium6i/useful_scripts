# -*- coding: utf-8 -*-
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
                - Added support for YOLOv8 classify model.
                - Modified image preprocessing to proportional scaling and padding.
                - Fixed issue with GPU inference failures.
    2024-06-07: - Added support for YOLOv10 (detect).
    2024-07-02: - Adjusted code structure.
                - Added support for YOLOv8 pose model.
    2024-07-03: - Adjusted code structure.
    2024-07-04: - Adjusted code structure.
    2024-07-09: - Change bbox format from xywh to xyxy.
    2024-07-10: - Bug fixes.
    2024-07-11: - Bug fixes.
    2024-07-12: - Bug fixes.
    2024-07-16: - Optimized output format and visualizd image.
    2024-07-18: - Bug fixes.
    2024-07-24: - Added support for YOLOv8 obb model.
    2024-07-30: - Adjusted code structure.
    2024-08-12: - Bug fixes.
    2024-08-14: - Added support for saving inference results as VOC format.
    2024-10-26: - Change np.int64 to np.int16 to reduce memory usage and possibly
                  improve performance.
    2024-11-30: - Minor performance improvements and code optimizations.

"""

import ast
import os
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from onnxruntime import InferenceSession, SessionOptions
from PIL import Image, ImageDraw, ImageFont
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
        self.colorset = [
            (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
            (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
            (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
            ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
            (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
            (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
            (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
            (217, 134, 177), (133, 111, 102), 
        ]

        self.initialize_model(model_path)

    def initialize_model(self, model_path):
        """
        Load and initialize the ONNX model.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        session_options = SessionOptions()

        cpu_nums = os.cpu_count()
        if self.thread_num != -1 and 1 <= self.thread_num <= cpu_nums:
            session_options.intra_op_num_threads = self.thread_num

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = InferenceSession(model_path, session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        model_info = self.session.get_modelmeta().custom_metadata_map
        self.labels = ast.literal_eval(model_info['names'])  # {0: 'cat', ...}
        self.imgsz = ast.literal_eval(model_info['imgsz'])  # [640, 640]
        self.task = model_info['task']  # detect, classify, pose, obb
        self.kpt_shape = (
            ast.literal_eval(model_info['kpt_shape'])
            if 'kpt_shape' in model_info
            else None
        )

    def preprocess(self, im):
        """
        Resize and pad the input image for model inference.

        Args:
            im (np.ndarray): Image array.

        Returns:
            im (np.ndarray): Preprocessed image data.
        """
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
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

        im = im.astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))  # Channel first
        im = np.expand_dims(im, axis=0).astype(np.float32)

        return im

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from xywh to xyxy.

        Args:
            x (np.ndarray): Box in [x, y, w, h] format.

        Returns:
            y (list): Converted box in [(x1, y1), (x2, y2)] format.
        """
        y = np.empty_like(x)  # faster than clone/copy
        xy = x[:2]  # centers
        wh = x[2:] / 2  # half width-height
        y[:2] = xy - wh  # top left xy
        y[2:] = xy + wh  # bottom right xy
        y = y.astype(np.int16)
        y[y < 0] = 0
        y = list(zip(y[::2], y[1::2]))

        return y

    def xywhr2xyxyxyxy(self, x):
        """
        Convert Oriented Bounding Boxes (OBB) coordinates from xywhr to xyxyxyxy.
        Rotation values should be in radian from 0 to 90.

        Args:
            x (np.ndarray): Box in [x, y, w, h, r] format.

        Returns:
            y (list): Converted box in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] format.
        """
        ctr = x[:2]
        w, h, angle = x[2:]
        cos_value, sin_value = np.cos(angle), np.sin(angle)
        vec1 = [w / 2 * cos_value, w / 2 * sin_value]
        vec2 = [-h / 2 * sin_value, h / 2 * cos_value]

        # Using vectors to calculate the coordinates of each point
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        y = [tuple(pt) for pt in np.stack([pt1, pt2, pt3, pt4]).astype(np.int16)]

        return y

    def detect_postproc(self, outs):
        """
        Postprocess of detect model.

        Args:
            outs (np.ndarray): Inference results from the model.

        Returns:
            results (list): Filtered and formatted results.
                            [[id, name, score, box, kpt, obb], ...].
        """
        nc = len(self.labels)  # number of classes
        outs = np.transpose(outs)

        # Filter and reshape results
        max_scores = np.max(outs[:, 4:4 + nc], axis=1)
        class_ids = np.argmax(outs[:, 4:4 + nc], axis=1)
        valid_indices = np.where(max_scores >= self.conf_thres)
        zeros = np.zeros((len(valid_indices[0]), 1))

        raw_xywh = outs[:, :4][valid_indices]
        raw_r = outs[:, -1][valid_indices].reshape(-1, 1) if self.task == 'obb' else zeros
        raw_scores = max_scores[valid_indices].reshape(-1, 1)
        raw_ids = class_ids[valid_indices].reshape(-1, 1)
        raw_kpts = outs[:, 4 + nc:][valid_indices] if self.task == 'pose' else zeros

        outs = np.hstack((raw_xywh, raw_r, raw_scores, raw_ids, raw_kpts))  # [[x, y, w, h, r, score, class_id, kpts], ...]

        # Restore bboxes and keypoints to original size
        s, t, l = self.scale
        outs[:, :2] -= [l, t]
        outs[:, :4] /= s

        if self.task == 'pose':
            x_cols = np.arange(7, outs.shape[1], 2)
            y_cols = np.arange(8, outs.shape[1], 2)
            outs[:, x_cols] = (outs[:, x_cols] - l) / s
            outs[:, y_cols] = (outs[:, y_cols] - t) / s

        # Prepare data for NMS
        boxes = outs[:, :4]
        scores = outs[:, 5]
        class_ids = outs[:, 6]
        kpts = outs[:, 7:] if self.task == 'pose' else zeros
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        results = []
        for i in indices:
            class_id = int(class_ids[i])
            class_name = self.labels[class_id]
            score = round(float(scores[i]), 4)
            if self.task != 'obb':
                box = self.xywh2xyxy(boxes[i])
                obb = []
            else:
                box = self.xywhr2xyxyxyxy(outs[:, :5][i])
                box_r_list = outs[:, :5][i]
                obb = [int(num) for num in box_r_list[:4]] + [round(box_r_list[4], 4)]
            kpt = kpts[i].astype(np.int16) if self.task == 'pose' else []
            kpt = list(zip(kpt[::2], kpt[1::2]))

            results.append([class_id, class_name, score, box, kpt, obb])

        return results

    def postprocess(self, outs):
        """
        Postprocess the model inference results.

        Args:
            outs (list): Raw inference results from the model.

        Returns:
            results (list): Filtered and formatted results.
                            [[id, name, score, box, kpt, obb], ...].
        """
        outs = np.squeeze(outs)

        if self.task == 'classify':
            class_id = np.argmax(outs)
            class_name = self.labels[class_id]
            score = outs[class_id]
            results = [[class_id, class_name, score]]

        elif self.task in ['detect', 'pose', 'obb']:
            results = self.detect_postproc(outs)

        else:
            print('Unsupported task.')
            results = None

        return results

    def model_infer(self, im):
        """
        Model inference process.

        Args:
            im (np.ndarray): Image array.

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
            print(f'Inference time of {self.task.capitalize()} model: {dt * 1000:.2f} ms')
            return results
        else:
            return results, dt

    def draw_boxes(self, im, results):
        """
        Visualize boxes based on filtered results.

        Args:
            im (np.ndarray): Image array.
            results (list): Filtered results.

        Returns:
            im (np.ndarray): Visualized image.
        """
        # Convert OpenCV image array to PIL image object
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        im_w, im_h = image.size

        # Text height, text font and line width
        th = 12 if min(im_w, im_h) < 600 else int(max(im_w, im_h) * 0.015)
        font = ImageFont.truetype(self.font_path, th)
        lw = int(max(im_w, im_h) * 0.003) + 1

        for i in results:
            label_id, label_name, conf, boxes, kpts, obb = i
            label_color = self.colorset[label_id % len(self.colorset)]
            label_text = f'{label_name} {conf * 100:.2f}%'
            tw = int(draw.textlength(label_text, font=font))

            if obb:  # Obb task
                # Draw bbox
                draw.polygon(boxes, outline=label_color, width=lw)

                # Draw label
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = boxes
                draw.rectangle(
                    [(x1, y1 - th - lw * 2), (x1 + tw + lw, y1)], 
                    fill=label_color, 
                    width=lw)
                draw.text(
                    (x1 + lw, y1 - th - lw), 
                    label_text, 
                    font=font, 
                    fill=(255, 255, 255))

            else:  # Not obb task
                # Draw bbox
                draw.rectangle(boxes, outline=label_color, width=lw)

                # Draw label
                (xmin, ymin), (xmax, ymax) = boxes
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
                    [(x1, y1 - lw), (x2 + lw, y2)], 
                    fill=label_color, 
                    width=lw)
                draw.text(
                    (x1 + lw, y1), 
                    label_text, 
                    font=font, 
                    fill=(255, 255, 255))

                # Draw keypoints
                if kpts:
                    for idx, j in enumerate(kpts):
                        kpt_color = self.colorset[idx % len(self.colorset)]
                        # top-left and bottom-right, lw as radius
                        tl = (j[0] - lw, j[1] - lw)
                        bw = (j[0] + lw, j[1] + lw) 
                        draw.ellipse([tl, bw], fill=kpt_color)

        # Blend drawed image and src image for transparent labels
        drawed = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.addWeighted(drawed, 0.7, im, 0.3, 0, drawed)

        return drawed

    def create_xml(self, results, save_path):
        # Create the root element
        annotation = ET.Element('annotation')

        # Add child elements to the root
        folder = ET.SubElement(annotation, 'folder')
        filename = ET.SubElement(annotation, 'filename')
        path = ET.SubElement(annotation, 'path')
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        segmented = ET.SubElement(annotation, 'segmented')

        file_name = os.path.basename(self.img_path) if self.img_path else 'test_image.jpg'
        folder.text = 'JPEGImages'
        filename.text = file_name
        path.text = '/work/' + file_name
        database.text = 'Unknown'
        height.text, width.text, depth.text = map(str, self.im_array.shape)
        segmented.text = '0'

        # Sort the results list by the name field (result[1])
        results_sorted = sorted(results, key=lambda x: x[1])

        # Iterate through the data list and create object elements for each item
        for result in results_sorted:
            obj = ET.SubElement(annotation, 'object')
            name = ET.SubElement(obj, 'name')
            pose = ET.SubElement(obj, 'pose')
            truncated = ET.SubElement(obj, 'truncated')
            difficult = ET.SubElement(obj, 'difficult')

            name.text = result[1]
            pose.text = 'Unspecified'
            truncated.text = '0'
            difficult.text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')

            (x1, y1), (x2, y2) = result[3]
            xmin.text, ymin.text, xmax.text, ymax.text = map(str, [x1, y1, x2, y2])

        tree = ET.ElementTree(annotation)
        ET.indent(tree, space='\t', level=0)
        tree.write(save_path, encoding='utf-8')

    def save_results(self, im, results):
        """
        Create a save folder and save the visualized images or annotation
        files. When input is np.ndarray, set the code file as an image file.

        Args:
            im (np.ndarray): Image array.
            results (list): Inference results after postprocessing.
        """
        if not (self.visualize or self.save_xml):
            return

        if not self.save_dir:
            # os.path.abspath(__file__) is the path of current code file.
            current_file = self.img_path if self.img_path else os.path.abspath(__file__)
            current_file_dir = os.path.dirname(current_file)
            self.save_dir = os.path.join(current_file_dir, 'Infer_Results')

        elif self.save_dir and os.path.isfile(self.save_dir):
            self.save_dir = os.path.join(os.path.dirname(self.save_dir), 'Infer_Results')

        if self.visualize:
            self.visualize_dir = os.path.join(self.save_dir, 'Visualized')
            os.makedirs(self.visualize_dir, exist_ok=True)
            save_name = os.path.basename(self.img_path) if self.img_path else 'visualized.jpg'
            save_path = os.path.join(self.visualize_dir, save_name)
            cv2.imwrite(save_path, self.draw_boxes(im, results))

            if self.im_count == 1:
                print(f'Visualized result saved at: {save_path}')

        if self.save_xml and self.task == 'detect':
            self.xmls_dir = os.path.join(self.save_dir, 'Annotations')
            os.makedirs(self.xmls_dir, exist_ok=True)
            if self.img_path:
                base, ext = os.path.splitext(os.path.basename(self.img_path))
                save_name = base + '.xml'
            else:
                save_name = 'annotation.xml'
            save_path = os.path.join(self.xmls_dir, save_name)
            self.create_xml(results, save_path)

            if self.im_count == 1:
                print(f'Annotation result saved at: {save_path}')

    def predict(self, im, save_dir=None, visualize=False, font_path=None, save_xml=False):
        """
        Run model inference on the input image.

        Args:
            im (np.ndarray|str): Image array or image(s) directory.

        Returns:
            results (list): Inference results after postprocessing.
        """
        self.save_dir = save_dir
        self.visualize = visualize if self.task != 'classify' else False
        self.font_path = font_path
        self.save_xml = save_xml
        if self.save_xml and self.task != 'detect':
            print('save_xml only support detect model.')

        # numpy array as input
        if isinstance(im, np.ndarray):
            self.im_count = 1
            self.img_path = False
            self.im_array = im
            results = self.model_infer(self.im_array)
            self.save_results(self.im_array, results)

        # path as input
        elif isinstance(im, str):
            if os.path.isfile(im):  # Single image
                self.im_count = 1
                self.img_path = im
                self.im_array = cv2.imread(im)
                results = self.model_infer(self.im_array)
                self.save_results(self.im_array, results)

            else:  # Directory of images
                self.im_count = -1
                img_list = [i for i in os.listdir(im) 
                    if os.path.isfile(os.path.join(im, i))]
                t = 0
                for img in tqdm(img_list, total=len(img_list), desc='Processing'):
                    img_path = os.path.join(im, img)
                    self.img_path = img_path
                    self.im_array = cv2.imread(img_path)
                    results, dt = self.model_infer(self.im_array)
                    self.save_results(self.im_array, results)
                    t += dt

                print(f'Inference time:\n    Total: {t * 1000:.2f} ms')
                print(f'    Avg: {t * 1000 / len(img_list):.2f} ms')
                if self.visualize:
                    print(f'Visualized results saved at: {self.visualize_dir}')
                if self.save_xml and self.task == 'detect':
                    print(f'Annotation result saved at: {self.xmls_dir}')
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

    def xyxy2xywh(self, x):
        """
        Convert bounding box coordinates from xyxy to xywh.

        Args:
            x (np.ndarray): Box in [x1, y1, x2, y2] format

        Returns:
            y (np.ndarray): Converted box in [x, y, w, h] format.
        """
        y = np.empty_like(x)  # faster than clone/copy
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height

        return y

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
        outs[:, :4] = self.xyxy2xywh(outs[:, :4])

        # Restore bboxes to original size
        s, t, l = self.scale
        outs[:, :2] -= [l, t]
        outs[:, :4] /= s

        boxes = outs[:, :4]
        scores = outs[:, -2]
        class_ids = outs[:, -1]

        results = []
        for i in indices:
            class_id = int(class_ids[i])
            class_name = self.labels[class_id]
            score = round(float(scores[i]), 4)
            box = self.xywh2xyxy(boxes[i])

            results.append([class_id, class_name, score, box, [], []])

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
        visualize=False,
        font_path=font_path,
        save_xml=True  # Only support detect model
        )
