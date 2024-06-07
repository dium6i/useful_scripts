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
                - Adapted inference for YOLOv8 classification model.
                - Modified image preprocessing to proportional scaling and padding.
                - Fixed issue with GPU inference failures.
    2024-05-23: - Use all CPUs for model inference by default.
    2024-06-07: - Added support for YOLOv10.

"""

import ast
import os

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def non_max_suppression(boxes, scores, iou_thresh):
    """
    Perform Non-Maximum Suppression to filter out overlapping bounding boxes.

    Args:
        boxes (list): Bounding boxes in the format [[x1, y1, x2, y2], ...].
        scores (list): Confidence scores corresponding to each bounding box.
        iou_thresh (float): Threshold for Intersection over Union.

    Returns:
        keep_indices (list): Indices to keep after NMS.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by their scores in descending order
    sorted_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )

    keep_indices = []
    while len(sorted_indices) > 0:
        # Pick the box with the highest confidence score
        max_index = sorted_indices[0]
        keep_indices.append(max_index)

        # Compute IoU between the picked box and all other boxes
        selected_box = boxes[max_index]
        other_boxes = [boxes[i] for i in sorted_indices[1:]]
        iou_scores = [
            calculate_iou(selected_box, other_box)
            for other_box in other_boxes
        ]

        # Filter out boxes with IoU greater than threshold
        filtered_indices = [
            i for i, iou in zip(sorted_indices[1:], iou_scores)
            if iou <= iou_thresh
        ]
        sorted_indices = filtered_indices

    return keep_indices


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): Bounding box in the format [x1, y1, x2, y2].
        box2 (list): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        iou: Intersection over Union (IoU) score.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the coordinates of the intersection rectangle
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    # Calculate intersection area
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * \
        max(0, intersection_y2 - intersection_y1 + 1)

    # Calculate areas of each bounding box
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    # Calculate Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def load_model(model_path, thread_num=-1):
    """
    Model initialization.

    Args:
        model_path (str): Model path.
        thread_num (int): Number of threads to use.

    Returns:
        session: Onnx inference session.
    """
    if thread_num == -1:  # Use all cpus
        thread_num = os.cpu_count()

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = thread_num

    try:
        session = ort.InferenceSession(
            model_path, session_options, providers=providers)
    except Exception as e:
        print(f"GPU inference failed: {e}. Falling back to CPU.")
        session = ort.InferenceSession(
            model_path, session_options, providers=['CPUExecutionProvider'])

    return session


def preprocess(img, imgsz):
    """
    Resize image and pad it with gray pixels for model inference.

    Args:
        img (numpy.ndarray): Image array.
        imgsz (int): Model input size. Like 640, 416, etc.

    Returns:
        img (numpy.ndarray): Preprocessed image data.
        tuple: Scale info used to restore bbox.
    """
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    ratio = min(imgsz / h, imgsz / w)

    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (new_w, new_h))

    dh = imgsz - new_h  # Total added pixels in height.
    dw = imgsz - new_w  # Total added pixels in width.
    t, b = dh // 2, dh - (dh // 2)  # Added pixels in top and bottom.
    l, r = dw // 2, dw - (dw // 2)  # Added pixels in left and right.

    color = [127, 127, 127]
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=color)

    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Channel first
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img, (ratio, t, l)


def postprocess(outs, labels, task, scale, conf=0.5, iou=0.6):
    """
    Postprocess inference results.

    Args:
        outs (list): Raw inference results.
        conf (float): Confidence threshold to filter results.
        iou (float): Intersection Over Union (IoU) threshold for Non-Maximum
                     Suppression (NMS).

    Returns:
        results (list): Filtered results in the format of
                        [[id, name, score, xmin, ymin, w, h], ...].
    """
    if task == 'classify':
        class_id = np.argmax(outs[0][0])
        class_name = labels[class_id]
        score = outs[0][0][class_id]

        return [class_id, class_name, score]

    elif task == 'detect':
        outs = np.squeeze(outs)
        if outs.shape[-1] != 6:  # v8, shape: (4 + Number of categories, ...)
            outs = np.transpose(outs)
            boxes, scores, class_ids = [], [], []

            max_scores = np.max(outs[:, 4:], axis=1)
            class_ids = np.argmax(outs[:, 4:], axis=1)
            valid_indices = np.where(max_scores >= conf)
            outs = np.hstack((
                outs[:, :4][valid_indices],
                max_scores[valid_indices].reshape(-1, 1),
                class_ids[valid_indices].reshape(-1, 1),
            ))  # [[x, y, w, h, score, class_id], ...]

            # Restore bboxes to original size
            x, y = outs[:, 0].copy(), outs[:, 1].copy()
            w, h = outs[:, 2].copy(), outs[:, 3].copy()
            s, t, l = scale
            outs[:, 0] = ((x - w / 2) - l) / s
            outs[:, 1] = ((y - h / 2) - t) / s
            outs[:, 2] = w / s
            outs[:, 3] = h / s

            # Use xyxy to NMS
            # outs[:, 2] = ((x + w / 2) - l) / s
            # outs[:, 3] = ((y + h / 2) - t) / s

            outs[:, :4] = outs[:, :4].astype(int)

            boxes = outs[:, :4]
            scores = outs[:, -2]
            class_ids = outs[:, -1]

            # Use xyxy to NMS
            # indices = non_max_suppression(boxes, scores, iou)
            # Use xywh to NMS
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)

            results = [[int(class_ids[i]),
                        labels[int(class_ids[i])],
                        round(float(scores[i]), 4),
                        *[int(j) for j in boxes[i]]] for i in indices]

            return results

        else:  # v10, shape: (300, 6)
            scores = outs[:, 4]
            indices = np.where(scores > conf)[0]
            outs = outs[indices, :]

            # From xyxy to xywh
            x1, y1 = outs[:, 0].copy(), outs[:, 1].copy()
            x2, y2 = outs[:, 2].copy(), outs[:, 3].copy()
            w, h = x2 - x1, y2 - y1
            x, y = x1 + w / 2, y1 + h / 2

            # Restore bboxes to original size
            s, t, l = scale
            outs[:, 0] = ((x - w / 2) - l) / s
            outs[:, 1] = ((y - h / 2) - t) / s
            outs[:, 2] = w / s
            outs[:, 3] = h / s

            boxes = outs[:, :4]
            scores = outs[:, -2]
            class_ids = outs[:, -1]

            results = [[int(class_ids[i]),
                        labels[int(class_ids[i])],
                        round(float(scores[i]), 4),
                        *[int(j) for j in boxes[i]]] for i in range(len(outs))]

            return results

    else:
        print('Unsupported task.')

        return None


def model_predict(session, img):
    """
    Main process to predict an image.

    Args:
        session: Onnx inference session.
        img (numpy.ndarray): Image array.

    Returns:
        results (list): Final results in the format of
                        [[id, name, score, xmin, ymin, w, h], ...].
    """
    model_info = session.get_modelmeta().custom_metadata_map
    labels = ast.literal_eval(model_info['names'])  # {0: 'cat', 1: 'dog', ...}
    imgsz = ast.literal_eval(model_info['imgsz'])  # [640, 640], [416, 416]
    task = model_info['task']  # detect, classify

    image_data, scale = preprocess(img, imgsz[0])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_data})
    results = postprocess(outputs, labels, task, scale)

    return results


if __name__ == '__main__':
    image_dir = 'path/of/image/directory'  # Image file or directory
    model_path = 'path/of/model'  # Path of model
    model = load_model(model_path)

    if os.path.isfile(image_dir):
        img = cv2.imread(image_dir)
        results = model_predict(model, img)
        print(results)

    else:
        for image in tqdm(os.listdir(image_dir), desc='Processing'):
            img = cv2.imread(os.path.join(image_dir, image))
            results = model_predict(model, img)
