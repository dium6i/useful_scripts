'''
Author: Wei Qin
Date: 2023-12-15
Description:
    Find images with specific labels in VOC dataset. 
Update Log:
    2023-12-15: File created.

'''

import os
import xml.etree.ElementTree as ET


data = 'path/of/voc/dataset'
imgs = os.path.join(data, 'JPEGImages')
xmls = os.path.join(data, 'Annotations')

finds = ['cat']  # Labels hope to find.
i = 0
f = 10  # Number of images hope to find.
found = []

skip = False
copy = True  # Copy found images to other directory.
copy_to = 'path/of/saving'

for img in os.listdir(imgs):
    if i == f:  # Found enough images.
        break

    base_name = '.'.join(img.split('.')[:-1])
    xml_path = os.path.join(xmls, base_name + '.xml')
    if not os.path.exists(xml_path):
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        if skip:  # Current image has been counted.
            continue

        if obj.find('name').text in finds:
            i += 1
            found.append(img)
            skip = True  # Skip current image.

            print(base_name)

    skip = False

if copy:
    import shutil

    copied_imgs = os.path.join(copy_to, 'JPEGImages')
    if not os.path.exists(copied_imgs):
        os.mkdir(copied_imgs)
    copied_xmls = os.path.join(copy_to, 'Annotations')
    if not os.path.exists(copied_xmls):
        os.mkdir(copied_xmls)

    for img in found:
        base_name = '.'.join(img.split('.')[:-1])
        xml_path = os.path.join(xmls, base_name + '.xml')

        shutil.copyfile(os.path.join(imgs, img), 
        	os.path.join(copied_imgs, img))
        shutil.copyfile(xml_path,
            os.path.join(copied_xmls, base_name + '.xml'))
