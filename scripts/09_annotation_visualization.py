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

'''

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

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
        xmin, ymin, xmax, ymax = (int(bndbox.find(pos).text) for pos in [
            'xmin', 'ymin', 'xmax', 'ymax'])
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


def draw_bbox(params, image, annotations, include=None, exclude=None):
    '''
    Draw bbox and label on image and save it.

    Args:
        params (dict): Parameters.
        image (str): Image filename.
        annotations (dict): Results of all annotations.
        include (list): Labels to include.
        exclude (list): Labels to exclude.

    Returns:
        None.
    '''
    snippet = image.split('.')
    base = '.'.join(snippet[:-1])  # Avoid filename like abc.def.jpg
    xml = base + '.xml'

    colors = [
        (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
        (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
        (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
        ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
        (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
        (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
        (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
        (217, 134, 177), (133, 111, 102), 
    ]

    img_path = os.path.join(params['imgs'], image)
    im = Image.open(img_path)

    if im.mode != 'RGB':  # Convert grayscale image to RGB mode
        im = im.convert('RGB')
    draw = ImageDraw.Draw(im)

    w, h = im.size
    lw = int(max(w, h) * 0.003) + 1

    results = annotations[xml]
    for i in results:
        label, xmin, ymin, xmax, ymax = i

        if include and label not in include:
            continue
        elif exclude and label in exclude:
            continue

        color_id = params['labels'].index(label)
        color = colors[label_id % len(colors)]

        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=color, width=lw)

        if params['show_label']:
            # Draw label
            if min(w, h) < 600:
                th = 12
            else:
                th = int(max(w, h) * 0.015)

            font = ImageFont.truetype(params['font_dir'], th)
            tw = draw.textlength(label, font=font)

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
                label,
                font=font,
                fill=(255, 255, 255))

    im.save(os.path.join(params['save'], image), quality=95)


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
        'font_dir': 'path/of/font',  # Font path
        'include': None,  # Labels hoping to visualize, None for all labels. Prior than exclude.
        'exclude': None,  # Labels not hoping to visualize
        'show_label': True,  # Whether to show labels of bboxes while visualization
    }

    run(params)
