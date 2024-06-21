"""
Author: Wei Qin
Date: 2024-06-12
Description:
    Convert VOC labels to YOLO format.
Update Log:
    2024-06-12: - File created.
    2024-06-21: - Deleted the function of reading label files.

"""

import os
import xml.etree.ElementTree as ET

import yaml


def xyxy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh

    return (x, y, w, h)


def voc2yolo(labels, xmls_dir, save_dir):
    xmls = os.listdir(xmls_dir)
    print(f'{len(xmls)} XML files to be converted.\nConverting... ')

    for xml in xmls:
        xml_path = os.path.join(xmls_dir, xml)
        txt_path = os.path.join(save_dir, os.path.splitext(xml)[0] + '.txt')

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
    labels = read_labels(label_path)
    xmls_dir = '/path/to/xmls/folder'
    save_dir = '/path/to/save/txt/files'
    os.makedirs(save_dir, exist_ok=True)

    voc2yolo(labels, xmls_dir, save_dir)
