'''
Author: Wei Qin
Date: 2023-12-22
Description:
    Delete images(labels) without corresponding labels(images). 
    Only for VOC dataset. 
Update Log:
    2023-12-22: File created.

'''

import os


data = 'path/of/dataset'

xmls = os.path.join(data, 'Annotations')
imgs = os.path.join(data, 'JPEGImages')
xmls_del = os.listdir(xmls)
imgs_del = os.listdir(imgs)

for img in imgs_del[:]:
    snippet = img.split('.')
    base = '.'.join(snippet[:-1])

    xml = base + '.xml'
    if xml in xmls_del:
        xmls_del.remove(xml)
        imgs_del.remove(img)

for i in xmls_del:
    os.remove(os.path.join(xmls, i))
for i in imgs_del:
    os.remove(os.path.join(imgs, i))
print(f'{len(xmls_del)} label(s)\n{len(imgs_del)} image(s)\nhave been deleted.')
