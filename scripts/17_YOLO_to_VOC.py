"""
Author: Wei Qin
Date: 2024-06-24
Description:
    Convert YOLO labels to VOC format.
Update Log:
    2024-06-24: - File created.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm


class Converter:
    def __init__(self, labels, imgs, txts, save):
        os.makedirs(save, exist_ok=True)
        self.labels = labels
        self.imgs = imgs
        self.txts = txts
        self.save = save

    def convert_image(self, img):
        txt = os.path.splitext(img)[0] + '.txt'
        txt_path = os.path.join(self.txts, txt)
        img_path = os.path.join(self.imgs, img)

        im = cv2.imread(img_path)
        img_h, img_w, _ = im.shape

        # Create VOC XML structure
        annotation = ET.Element('annotation')

        folder = ET.SubElement(annotation, 'folder')
        filename = ET.SubElement(annotation, 'filename')
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        segmented = ET.SubElement(annotation, 'segmented')

        folder.text = os.path.basename(self.imgs)
        filename.text = img
        width.text = str(img_w)
        height.text = str(img_h)
        depth.text = '3'
        segmented.text = '0'

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            # xywh to xyxy
            xmin = int((x_center - bbox_width / 2) * img_w)
            ymin = int((y_center - bbox_height / 2) * img_h)
            xmax = int((x_center + bbox_width / 2) * img_w)
            ymax = int((y_center + bbox_height / 2) * img_h)

            object_elem = ET.SubElement(annotation, 'object')
            name = ET.SubElement(object_elem, 'name')
            pose = ET.SubElement(object_elem, 'pose')
            truncated = ET.SubElement(object_elem, 'truncated')
            difficult = ET.SubElement(object_elem, 'difficult')
            bndbox = ET.SubElement(object_elem, 'bndbox')
            xmin_elem = ET.SubElement(bndbox, 'xmin')
            ymin_elem = ET.SubElement(bndbox, 'ymin')
            xmax_elem = ET.SubElement(bndbox, 'xmax')
            ymax_elem = ET.SubElement(bndbox, 'ymax')

            name.text = self.labels[class_id]
            pose.text = 'Unspecified'
            truncated.text = '0'
            difficult.text = '0'
            xmin_elem.text = str(xmin)
            ymin_elem.text = str(ymin)
            xmax_elem.text = str(xmax)
            ymax_elem.text = str(ymax)

        # Write XML to file
        xml_filename = os.path.splitext(img)[0] + '.xml'
        xml_path = os.path.join(self.save, xml_filename)
        tree = ET.ElementTree(annotation)
        ET.indent(tree, space="\t", level=0)  # Python >= 3.9
        tree.write(xml_path)


def yolo_to_voc(labels, imgs, txts, save):
    converter = Converter(labels, imgs, txts, save)
    img_list = [
        i for i in os.listdir(imgs) if os.path.isfile(os.path.join(imgs, i))
    ]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(converter.convert_image, img) for img in img_list
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc='Progress'
        ):
            pass


if __name__ == '__main__':
    labels = ['cat', 'dog', 'person']
    imgs = 'path/of/images'
    txts = 'path/of/labels'
    save = 'path/to/save/xmls'

    yolo_to_voc(labels, imgs, txts, save)
