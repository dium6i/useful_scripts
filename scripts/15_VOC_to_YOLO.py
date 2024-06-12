"""
Author: Wei Qin
Date: 2024-06-12
Description:
    Convert VOC labels to YOLO format.
Update Log:
    2024-06-12: - File created.

"""

import os
import xml.etree.ElementTree as ET

import yaml


def read_labels(label_path):
    if isinstance(label_path, list):
        return label_path

    if label_path.endswith('.txt'):
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [label.rstrip('\n') for label in labels]
        return labels

    elif label_path.endswith('.yml') or label_path.endswith('.yaml'):
        with open(label_path, 'r') as f:
            labels = yaml.safe_load(f)['label_list']
        return labels

    else:
        return None
        print('Unsupported input type.')


def xyxy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh

    return (x, y, w, h)


def voc2yolo(data_path, labels):
    if not labels:  # No valid labels
        exit()

    # Create a folder(labels) to store the .txt files
    label_save = os.path.join(os.path.dirname(data_path), 'labels')
    if not os.path.exists(label_save):
        os.mkdir(label_save)

    xmls = os.listdir(data_path)
    print(f'{len(xmls)} XML files to be converted.\nConverting... ')

    for xml in xmls:
        xml_path = os.path.join(data_path, xml)
        txt_path = os.path.join(label_save, xml.replace('.xml', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as f:

            # Parsing XML files
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                cls_id = labels.index(cls_name)
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(i).text)
                       for i in ['xmin', 'ymin', 'xmax', 'ymax']]

                bbox = xyxy2xywh((w, h), box)
                f.write(f'{cls_id} ' + ' '.join(str(x) for x in bbox) + '\n')

    print('Convertion complete.')


if __name__ == '__main__':
    data_path = '/path/to/xmls/folder'
    label_path = ['cat', 'dog', 'person']  # labels list or path of labels file (txt or yml/yaml)
    labels = read_labels(label_path)
    voc2yolo(data_path, labels)
