'''
Author: Wei Qin
Date: 2024-02-23
Description:
    Visualize VOC annotations to check whether wrong labeling exists.
Update Log:
    2024-02-23: - File created.
                - Optimized the use of multi-threading and the effect
                  of visualization.
                - Fixed a bug where unlabeled images were not saved.
    2024-02-27: - Added new features and optimized code structure.
    2024-04-25: - Depending on whether the font_dir is provided, PIL
                  and OpenCV are used respectively for annotation
                  visualization.

'''

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import xml.etree.ElementTree as ET


def reprocess_params(params):
    '''
    Reprocess parameters.

    Args:
        params (dict): Parameters.

    Returns:
        params (dict): Parameters.
    '''
    params['imgs'] = os.path.join(params['data_dir'], params['imgs'])
    params['xmls'] = os.path.join(params['data_dir'], params['xmls'])
    params['save'] = os.path.join(params['data_dir'], params['save'])
    if not os.path.exists(params['save']):
        os.mkdir(params['save'])

    return params


def read_xml(xml, xml_path):
    '''
    Single XML file parsing process.

    Args:
        xml (str): XML file name.
        xml_path (str): Path of XML file.

    Returns:
        xml (str): XML file name.
        results (list): Result of labelings.
    '''
    results = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = (
            int(bndbox.find(i).text) for i in ['xmin', 'ymin', 'xmax', 'ymax'])
        results.append([label, xmin, ymin, xmax, ymax])

    return xml, results


def read_xmls(xmls):
    '''
    Read all XML files to extract annotations.

    Args:
        xmls (str): Path of XML directory.

    Returns:
        annotations (dict): Results of all annotations.
    '''
    annotations = {}
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(read_xml, xml, os.path.join(xmls, xml))
            for xml in os.listdir(xmls)
        ]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Parsing XML files'):
            k, v = future.result()
            annotations[k] = v

    return annotations


def get_labels(annotations):
    '''
    Get labels from annotations.

    Args:
        annotations (dict): Results of all annotations.

    Returns:
        labels (list): Labels in dataset.
    '''
    labels = []
    for k, v in annotations.items():
        labels.extend([i[0] for i in v])

    return sorted(set(labels))


def label_adjustment(w, xmin, ymin, tw, th, lw):
    '''
    Adjust the position of label to prevent incomplete labels.

    Args:
        w (int): Image width.
        xmin, ymin (int): Upper left Coordinates of bbox.
        tw, th (int): Label text width and height.
        lw (int): Line width

    Returns:
        x1, y1, x2, y2 (int): Adjusted label coordinates.
    '''
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

    return x1, y1, x2, y2


def draw_PIL(img_path, params, results, include=None, exclude=None):
    '''
    Draw and save using PIL.

    Args:
        img_path (str): Path of image file.
        params (dict): Parameters.
        results (list): Annotations of this image.
        include (list): Labels to include. Prior than exclude.
        exclude (list): Labels to exclude.

    Returns:
        None.
    '''
    im = Image.open(img_path)
    w, h = im.size
    lw = int(max(w, h) * 0.003) + 1

    if im.mode != 'RGB':  # Convert grayscale image to RGB mode
        im = im.convert('RGB')
    draw = ImageDraw.Draw(im)

    for i in results:
        label, xmin, ymin, xmax, ymax = i

        if include and label not in include:
            continue
        elif exclude and label in exclude:
            continue

        label_id = params['labels'].index(label)
        color = params['colors'][label_id % len(params['colors'])]

        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=color, width=lw)

        if params['show_label']:
            if min(w, h) < 600:
                th = 12
            else:
                th = int(max(w, h) * 0.015)

            font = ImageFont.truetype(params['font_dir'], th)
            tw = int(draw.textlength(label, font=font))

            x1, y1, x2, y2 = label_adjustment(w, xmin, ymin, tw, th, lw)

            draw.rectangle(
                [(x1, y1), (x2 + lw, y2)],
                fill=color,
                width=lw)
            draw.text(
                (x1 + lw, y1 + lw),
                label,
                font=font,
                fill=(255, 255, 255))

    filename = os.path.basename(img_path)
    im.save(os.path.join(params['save'], filename), quality=95)


def draw_cv(img_path, params, results, include=None, exclude=None):
    '''
    Draw and save using OpenCV.

    Args:
        img_path (str): Path of image file.
        params (dict): Parameters.
        results (list): Annotations of this image.
        include (list): Labels to include. Prior than exclude.
        exclude (list): Labels to exclude.

    Returns:
        None.
    '''
    font_scale = 0.5
    font_thickness = 1

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    lw = int(max(w, h) * 0.003) + 1

    for i in results:
        label, xmin, ymin, xmax, ymax = i

        if include and label not in include:
            continue
        elif exclude and label in exclude:
            continue

        label_id = params['labels'].index(label)
        color = params['colors'][label_id % len(params['colors'])]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=lw)

        if params['show_label']:
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            x1, y1, x2, y2 = label_adjustment(w, xmin, ymin, tw, th, lw)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.putText(
                img, label, (x1 + lw, y2), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), font_thickness)

    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(params['save'], filename), img)


def draw_bbox(params, image, annotations, include=None, exclude=None):
    '''
    Draw bbox and label on image and save it.

    Args:
        params (dict): Parameters.
        image (str): Image filename.
        annotations (dict): Results of all annotations.
        include (list): Labels to include. Prior than exclude.
        exclude (list): Labels to exclude.

    Returns:
        None.
    '''
    img_path = os.path.join(params['imgs'], image)
    base, ext = os.path.splitext(image)
    xml = base + '.xml'
    results = annotations[xml]

    if params['font_dir']:
        draw_PIL(img_path,
                 params,
                 results,
                 include=params['include'],
                 exclude=params['exclude'])
    else:
        draw_cv(img_path,
                params,
                results,
                include=params['include'],
                exclude=params['exclude'])


def run(params):
    '''
    Main process.

    Args:
        params (dict): Parameters.

    Returns:
        None.
    '''
    params = reprocess_params(params)

    annotations = read_xmls(params['xmls'])
    params['labels'] = get_labels(annotations)
    params['colors'] = [
        (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
        (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
        (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
        ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
        (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
        (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
        (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
        (217, 134, 177), (133, 111, 102), 
    ]

    with ThreadPoolExecutor() as executor:
        images = os.listdir(params['imgs'])
        futures = [
            executor.submit(
                draw_bbox,
                params,
                image,
                annotations,
                include=params['include'],
                exclude=params['exclude']) for image in images]
        for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Drawing'):
            pass


if __name__ == '__main__':
    # Setting parameters
    params = {
        'data_dir': 'path/of/dataset',  # Dataset directory
        'imgs': 'JPEGImages',  # Image folder
        'xmls': 'Annotations',  # Annotation folder
        'save': 'Visualization',  # Result folder
        'font_dir': 'path/of/font',  # Font path or None 
        'include': None,  # Labels hoping to visualize, None for all labels. Prior than exclude.
        'exclude': None,  # Labels not hoping to visualize
        'show_label': True,  # Whether to show labels of bboxes while visualization
    }

    run(params)
