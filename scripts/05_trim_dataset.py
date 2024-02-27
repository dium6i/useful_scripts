'''
Author: Wei Qin
Date: 2023-12-22
Description:
    Delete images(labels) without corresponding labels(images).
    Only for VOC dataset.
Update Log:
    2023-12-22: - File created.
    2024-02-27: - Optimized code structure for easier integration 
                  into other projects.
                - Improved the readability of output results.

'''

import os


def run(params):
    '''
    Main process.

    Args:
        params (dict): Parameters.

    Returns:
        None.
    '''
    xmls = os.path.join(params['data'], params['xmls'])
    imgs = os.path.join(params['data'], params['imgs'])
    xmls_del = os.listdir(xmls)
    imgs_del = os.listdir(imgs)

    for img in imgs_del[:]:
        snippet = img.split('.')
        base = '.'.join(snippet[:-1])

        xml = base + '.xml'
        if xml in xmls_del:
            xmls_del.remove(xml)
            imgs_del.remove(img)

    print(f'{len(xmls_del)} label(s) can be deleted:')
    print(xmls_del)
    print(f'{len(imgs_del)} image(s) can be deleted:')
    print(imgs_del)

    if params['delete']:
        for i in xmls_del:
            os.remove(os.path.join(xmls, i))
        for i in imgs_del:
            os.remove(os.path.join(imgs, i))

        print(f'\n{len(xmls_del)} label(s)')
        print(f'{len(imgs_del)} image(s)')
        print('have been deleted.')


if __name__ == '__main__':
    # Setting parameters
    params = {
        'data': r'D:\01_hzwq\15_jlx_dnb\02_dnb\temp',  # Dataset directory
        'imgs': 'JPEGImages',  # Image folder
        'xmls': 'Annotations',  # Label folder
        'show_info': True,  # Whether to show which files can be deleted
        'delete': True  # Whether to delete files
    }

    run(params)
