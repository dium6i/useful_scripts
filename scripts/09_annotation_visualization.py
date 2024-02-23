'''
Author: Wei Qin
Date: 2024-02-23
Description:
    Visualize VOC annotations to check whether wrong labeling exists.
Update Log:
    2024-02-23: - File created.
                - Optimized the use of multi-threading and the effect 
                  of visualization.

'''

from concurrent.futures import ProcessPoolExecutor, as_completed
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


def read_xml(xml_path):
    '''
    Read XML file to extract annotations.

    Args:
        xml_path (str): Path of XML file.

    Returns:
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

    return results


def draw_bbox(params, image):
    '''
    Draw bbox and label on image and save it.

    Args:
        params (dict): Parameters.
        image (str): Image filename.
        results (list): Annotations of this image.

    Returns:
        None.
    '''
    # Get annotations
    snippet = image.split('.')
    base = '.'.join(snippet[:-1])  # Avoid filename like abc.def.jpg
    xml = base + '.xml'
    xml_path = os.path.join(params['xmls'], xml)
    results = read_xml(xml_path)

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

    for i in results:
        label, xmin, ymin, xmax, ymax = i
        color = colors[params['labels'].index(label)]

        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=color, width=lw)

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

    # Process each image in parallel
    with ProcessPoolExecutor() as executor:
        images = os.listdir(params['imgs'])
        futures = [
            executor.submit(draw_bbox, params, image) for image in images
        ]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Drawing'):
            future.result()


if __name__ == '__main__':
    # Setting parameters
    params = {
        'labels': ['cat', 'dog', 'car'],
        'font_dir': 'path/of/font.ttf',
        'data_dir': 'path/of/dataset',
        'imgs': 'JPEGImages',
        'xmls': 'Annotations',
        'save': 'Visualization'
    }

    run(params)
