"""
Author: Wei Qin
Date: 2024-03-26
Description:
    Use the onnx model with the onnxruntime library for model inference.
Update Log:
    2024-03-26: - File created.
    2024-03-27: - Added functions: 
                    -- Whether to contain label name in the results.
                    -- Set inference threads.
    2024-03-29: - Use cv2.dnn.NMSBoxes() to replace the customized NMS method.

"""

import ast
import os

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def load_model(model_path, thread_num=1):
    """
    Model initialization.

    Args:
    - model_path: Model path.
    - thread_num: Number of threads to use.

    Returns:
    - session: Onnx inference session.
    """
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = thread_num
    # session_options.use_gpu = True
    session = ort.InferenceSession(model_path, session_options)

    return session


def preprocess(img, input_shape=(640, 640)):
    """
    Preprocess image.

    Args:
    - img: Read by OpenCV.
    - input_shape: Model input shape.

    Returns:
    - image_data: Preprocessed image data.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    return image_data


def postprocess(
        outs,
        img_shape,
        input_shape=(640, 640),
        labels=None,
        conf=0.5,
        iou=0.6):
    """
    Postprocess inference results.

    Args:
    - outs: Raw inference results.
    - img_shape: Image shape in the format (h, w, c).
    - input_shape: Model input shape.
    - labels: Model labels.
    - conf: Confidence threshold to filter results.
    - iou: Intersection Over Union (IoU) threshold for Non-Maximum 
           Suppression (NMS).

    Returns:
    - results: Filtered results in the format 
               [[class_id, score, xmin, ymin, w, h], ......] or
               [[class_id, class_name, score, xmin, ymin, w, h], ......].
    """
    outs = np.transpose(np.squeeze(outs))  # shape: (8400, 4 + labels number)
    boxes, scores, class_ids = [], [], []
    x_factor = img_shape[1] / input_shape[1]
    y_factor = img_shape[0] / input_shape[0]

    max_scores = np.max(outs[:, 4:], axis=1)
    class_ids = np.argmax(outs[:, 4:], axis=1)
    valid_indices = np.where(max_scores >= conf)
    outs = np.hstack((
        outs[:, :4][valid_indices],
        class_ids[valid_indices].reshape(-1, 1), 
        max_scores[valid_indices].reshape(-1, 1), 
        ))  # In the format [[x, y, w, h, class_id, score], ......]

    x, y, w, h = outs[:, 0].copy(), outs[:, 1].copy(), outs[:, 2].copy(), outs[:, 3].copy()
    outs[:, 0] = (x - w / 2) * x_factor
    outs[:, 1] = (y - h / 2) * y_factor
    outs[:, 2] = w * x_factor
    outs[:, 3] = h * y_factor
    outs[:, :4] = outs[:, :4].astype(int)

    boxes = outs[:, :4]
    scores = outs[:, -1]
    class_ids = outs[:, -2]
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)

    if labels:
        results = [[int(class_ids[i]), 
                    labels[int(class_ids[i])], 
                    float(scores[i]), 
                    *[int(j) for j in boxes[i]]] for i in indices]
    else:
        results = [[int(class_ids[i]), 
                    float(scores[i]), 
                    *[int(j) for j in boxes[i]]] for i in indices]

    return results


def model_predict(
        session,
        img,
        model_input_size=(640, 640),
        labels=None,
        with_label=False):
    """
    Main process to predict an image.

    Args:
    - session: Onnx inference session.
    - img: Read by OpenCV.
    - model_input_size: Model input size in the format (640, 640).
    - labels: Model labels.
    - with_label: Whether to contain label name in the results.

    Returns:
    - results: Final results in the format [[class_id, score, xmin, ymin, xmax, ymax], ......].
    """
    model_input_name = session.get_inputs()[0].name
    if with_label:
        labels_str = session.get_modelmeta().custom_metadata_map['names']
        labels = ast.literal_eval(labels_str)

    image_data = preprocess(img, model_input_size)
    outputs = session.run(None, {model_input_name: image_data})
    results = postprocess(outputs, img.shape, model_input_size, labels)

    return results


if __name__ == '__main__':
    image_dir = "path/of/image/directory"  # Image file or directory
    model_path = "path/of/model"  # Path of model
    model = load_model(model_path, thread_num=4)

    if os.path.isfile(image_dir):
        img = cv2.imread(image_dir)
        results = model_predict(model, img, with_label=True)
        print(results)
    else:
        for image in tqdm(os.listdir(image_dir)[:1], desc='Processing'):
            img = cv2.imread(os.path.join(image_dir, image))
            results = model_predict(model, img, with_label=True)
