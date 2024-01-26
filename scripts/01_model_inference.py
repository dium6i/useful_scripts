'''
Author: Wei Qin
Date: 2023-12-01
Description:
    Use detection model to inference, auto labeling or visualize results. 
    The current version is only suitable for model inference using FastDeploy.
Update Log:
    2023-12-01: - File created.
    2024-01-17: - Adjusted colorset.
    2024-01-18: - Optimized the label rendering in visualizations.
                - Renamed some variables to streamline the codebase.
    2024-01-25: - Added a new feature to toggle the visibility of labels 
                  during visualization, offering more customization to the 
                  user experience.
    2024-01-26: - Fixed a parsing error that occurred when file names 
                  contained multiple periods.

'''

import os
from xml.dom import minidom
import xml.etree.ElementTree as ET

import cv2
import fastdeploy as fd
from fastdeploy import ModelFormat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml


def read_labels(cfg_path):
    '''
    Read config file to load model labels.

    Args:
        cfg_path (string): Path of 'infer_cfg.yml' of inference model.

    Returns:
        data['label_list'] (list): Model labels. # ['preson', 'cat', 'dog']
    '''
    with open(cfg_path, 'r', encoding='utf-8') as f_cfg:
        data = yaml.safe_load(f_cfg)

    return data['label_list']


def filter_res(params, results, include=None, exclude=None):
    '''
    Filter and reorganize the detection results of model inference.

    Args:
        results (fd.DetResult): Results of model inference.
        include (list): Label ids to include.
        exclude (list): Label ids to exclude.

    Returns:
        filtered (list): Filtered and reorganized results like 
                         [[label_id, label_name, confidence, x1, y1, x2, y2], 
                          ...]
    '''
    filtered = []
    labels = params['labels']

    # Single threshold for all labels
    if type(params['confs']) == float:
        confs = [params['confs']] * len(labels)

    # Multi thresholds for each label
    elif type(params['confs']) == list:
        if len(params['confs']) == len(labels):
            confs = params['confs']
        else:
            confs = [params['confs'][0]] * len(labels)
            print('\nWarnning: \nThe number of thresholds provided in params does not match the number of labels. First threshold would be used for all labels.')

    # Invalid types
    else:
        confs = [0.5] * len(labels)
        print('\nWarnning: \nUnsupported types. Use default threshold (0.5) for all labels.')

    for i, j in enumerate(results.boxes):
        label_id = results.label_ids[i]
        if include and label_id not in include:
            continue
        elif exclude and label_id in exclude:
            continue

        # Filter result
        if results.scores[i] > confs[results.label_ids[i]]:
            filtered.append([
                results.label_ids[i],  # label id
                labels[results.label_ids[i]],  # label name
                round(results.scores[i], 4),  # confidence
                *results.boxes[i]  # bbox x1, y1, x2, y2
            ])

    # Integerize and rationalize coordinates
    for i in range(len(filtered)):
        filtered[i][3:] = [int(x) if int(
            x) >= 0 else 0 for x in filtered[i][3:]]

    return filtered


def draw_boxes(params, img, filtered):
    '''
    Visualize boxes based on filtered results.

    Args:
        params (dict): Parameters.
        img (numpy.ndarray): Read by OpenCV.
        filtered (list): Filtered results.

    Returns:
        img (numpy.ndarray): Visualized image.
    '''
    # Convert OpenCV image array to PIL image object
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # Determines the line width of bboxes
    w, h = image.size
    lw = int(max(w, h) * 0.003) + 1 # line width

    colorset = [
        [218, 179, 218], [138, 196, 208], [112, 112, 181], [255, 160, 100], 
        [106, 161, 115], [232, 190,  93], [211, 132, 252], [ 77, 190, 238], 
        [  0, 170, 128], [196, 100, 132], [205, 110,  70], [153, 153, 153], 
        [194, 194,  99], [ 74, 134, 255], [ 93,  93, 135], [140, 160,  77], 
        [255, 185, 155], [255, 107, 112], [165, 103, 190], [202, 202, 202], 
        [  0, 114, 189], [ 85, 170, 128], [ 60, 106, 117], [250, 118, 153], 
        [119, 172,  48], [171, 229, 232], [160,  85, 100], [223, 128,  83], 
        [217, 134, 177], [133, 111, 102], 
    ]

    for i in filtered:
        label_id, label_name, conf, xmin, ymin, xmax, ymax = i

        # Determine the color of the label
        color = tuple(colorset[label_id % len(colorset)])

        # Draw bbox
        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=color, width=lw)

        if params['show_label']:
            # Draw label
            text = f'{label_name} {conf * 100:.2f}%'
            if min(w, h) < 600:
                th = 12
            else:
                th = int(max(w, h) * 0.015)

            font = ImageFont.truetype(params['font_dir'], th)
            tw = draw.textlength(text, font=font)

            x1 = xmin
            y1 = ymin - th - lw
            x2 = xmin + tw + lw
            y2 = ymin

            if y1 < 10:  # Label-top out of image
                y1 = ymin
                y2 = ymin + th + lw

            if x2 > w:  # Label-right out of image
                x1 = w - tw - lw
                x2 = w

            draw.rectangle(
                [(x1, y1), (x2 + lw, y2)], 
                fill=color, 
                width=lw)
            draw.text(
                (x1 + lw, y1 + lw), 
                text, 
                font=font, 
                fill=(255, 255, 255))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def generate_XML(data, file_name, img_shape):
    '''
    Generate XML file based on filtered results.

    Args:
        data (list): Filtered and reorganized results.
        file_name (string): Filename.
        img_shape (tuple): Shape of an image. # (h, w, c)

    Returns:
        xmlstr (string): XML document as a string with proper indentation.
    '''
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

    folder.text = 'JPEGImages'
    filename.text = file_name
    path.text = '/work/' + file_name
    database.text = 'Unknown'
    width.text = str(img_shape[1])
    height.text = str(img_shape[0])
    depth.text = '3'
    segmented.text = '0'

    # Iterate through the data list and create object elements for each item
    for item in data:
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        pose = ET.SubElement(obj, 'pose')
        truncated = ET.SubElement(obj, 'truncated')
        difficult = ET.SubElement(obj, 'difficult')

        name.text = item[1]
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

        xmin.text = str(item[3])
        ymin.text = str(item[4])
        xmax.text = str(item[5])
        ymax.text = str(item[6])

    # Create and save the XML file with indentation
    xmlstr = minidom.parseString(
        ET.tostring(annotation)).toprettyxml(
        indent='    ')

    return xmlstr


def image_prediction(params, filename, model, im):
    '''
    Performs single-image processing, including model inference, 
    results reorganizing, XML file generation, and visualization 
    of detection results.

    Args:
        params (dict): Parameters.
        filename (string): Filename.
        model (fd.model): Model to predict images.
        im (numpy.ndarray): Read by OpenCV.

    Returns:
        None.
    '''
    snippet = filename.split('.')
    base = '.'.join(snippet[:-1])
    dot_ext = '.' + snippet[-1]  # .jpg
    xml = base + '.xml'

    results = model.predict(im)
    data = filter_res(
        params,
        results,
        include=params['include'],
        exclude=params['exclude'])

    if params['save_xml']:
        xmlstr = generate_XML(data, filename, im.shape)
        with open(os.path.join(params['xml_dir'], xml), 'w') as xml_file:
            xml_file.write(xmlstr)

    if params['visualize']:
        vis_img = draw_boxes(params, im, data)
        cv2.imwrite(os.path.join(params['visual_dir'], filename), vis_img)

    return data


def run(params, model):
    '''
    Main process.

    Args:
        params (dict): Parameters.
        model (fd.model): Model to predict images.

    Returns:
        None.
    '''
    if params['save_xml']:
        Annotations = os.path.join(params['save_dir'], 'Annotations')
        if not os.path.exists(Annotations):
            os.makedirs(Annotations)
        params['xml_dir'] = Annotations
    if params['visualize']:
        Visualizations = os.path.join(params['save_dir'], 'Visualizations')
        if not os.path.exists(Visualizations):
            os.makedirs(Visualizations)
        params['visual_dir'] = Visualizations

    # Single file prediction
    if os.path.isfile(params['img_dir']):
        filename = params['img_dir'].split('/')[-1]  # abc.jpg
        im = cv2.imread(params['img_dir'])

        data = image_prediction(params, filename, model, im)
        print('')
        print(data)

    # Batch prediction
    else:
        for filename in tqdm(os.listdir(params['img_dir']), desc='Processing'):
            if filename.startswith('.i'):  # Exclude hidden directory
                continue

            img_path = os.path.join(params['img_dir'], filename)
            im = cv2.imread(img_path)

            image_prediction(params, filename, model, im)


if __name__ == '__main__':
    # Setting parameters
    params = {
        'img_dir': 'path/of/image/directory',  # Image file or directory
        'save_dir': 'path/of/saving/directory',  # Result saved directory
        'font_dir': '/work/consolab.ttf',  # Visualize font directory
        'confs': 0.5, # Threshold for filtering results, 'float' for all labels, 'list' for each.
        'save_xml': True,  # Whether to save detection results to XML files
        'visualize': True,  # Whether to visualize detection results
        'show_label': True,  # Whether to show labels of bboxes while visualization
        'include': None,  # Label ids to include, like [1, 2, 7, 8, 9]
        'exclude': None  # Label ids to exclude
    }

    # Specify runtime options.
    runtime_option = fd.RuntimeOption()
    runtime_option.use_openvino_backend()
    # runtime_option.use_ort_backend()
    # runtime_option.use_paddle_infer_backend()

    # Using PaddlePaddle model (ModelFormat can left blank)
    model = fd.vision.detection.PPYOLOE(
        'path/of/model.pdmodel',
        'path/of/model.pdiparams',
        'path/of/infer_cfg.yml',
        runtime_option=runtime_option,
        model_format=ModelFormat.PADDLE)

    # Using ONNX model of YOLOv8(PyTorch)
    # model = fd.vision.detection.YOLOv8(
    #     'path/of/yolov8.onnx',
    #     runtime_option=runtime_option)

    # Read model labels
    cfg_path = 'path/of/infer_cfg.yml'
    labels = read_labels(cfg_path)

    # Specify model labels
    # labels = ['preson', 'cat', 'dog']
    
    params['labels'] = labels

    run(params, model)
