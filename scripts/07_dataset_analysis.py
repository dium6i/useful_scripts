'''
Author: Wei Qin, ChatGPT
Date: 2024-01-05
Description:
    Count the number of labels and the label size ratios of per class.
    Partially generated by ChatGPT and edited by me.
Update Log:
    2024-01-05: - File created.
                - Added / changed some comments and changed file name.
    2024-01-08: - Added a new feature for counting small objects.
    2024-01-11: - Modified some explanations of functions and comments
                  to improve overall code readability.
                - Fixed an issue where the same small object would be
                  counted by multiple size categories.
                - Added more details when counting small objects.
    2024-01-16: - Implemented a stacked bar chart feature to accurately
                  display category-wise proportions in the visualization
                  of small objects counting results.
                - Adjusted the color scheme.
    2024-01-17: - Adjusted the color scheme and optimized the
                  code architecture.
    2024-01-20: - Adjusted the color scheme and transitioned the criterion
                  for determining small labels from pixel-based measurements
                  to percentage of image area.
    2024-02-27: - Optimized code structure for easier integration
                  into other projects.
    2024-03-11: - Added functionality to display total count and individual
                  counts for each label.
    2024-04-07: - Added feature that automatically adjust the picture width
                - of label count based on the number of labels.
    2024-04-18: - Optimized code structure for easier integration
                  into other projects.

'''

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def process_xml(params, xml):
    '''
    Single xml file parsing process.

    Args:
        params (dict): Parameters.
        xml (string): File name or absolute path of XML file.

    Returns:
        lc (dict): Label counts.
                   lc = {'cat': 5, 'dog': 7}
        lr (dict): Label size ratios.
                   lr = {'cat': {'Height': [0.2, 0.6], 'Width': [0.4, 0.8]},
                         'dog': {'Height': [0.3, 0.7], 'Width': [0.5, 0.6]}}
        sc (dict): Small-object counts.
                   sc = {'10x10': ['image1.xml', 'image2.xml'],
                         '25x25': ['image3.xml', 'image4.xml'],
                         '50x50': ['image5.xml', 'image6.xml']}
    '''
    lc = defaultdict(int)
    lr = defaultdict(lambda: {'Height': [], 'Width': []})
    sc = {f'{x}x{x}': [] for x in params['split_range']}

    if isinstance(params['xmls'], str):  # Directory of annotations
        xml_path = os.path.join(params['xmls'], xml)
    elif isinstance(params['xmls'], list):  # List of annotation file
        xml_path = xml
    else:
        print("Unsupported input.(params['xmls'])")
        exit()

    # xml_path = os.path.join(params['xmls'], xml)

    # Ignore hidden directory
    if os.path.isdir(xml_path):
        return lc, lr, sc

    # Normal parsing process
    tree = ET.parse(xml_path)
    root = tree.getroot()

    get_image_size = params['size_ratio'] or params['small_object']

    if get_image_size:
        size = root.find('size')
        im_h = int(size.find('height').text)
        im_w = int(size.find('width').text)

    for obj in root.iter('object'):
        name = obj.find('name').text
        lc[name] += 1

        if get_image_size:
            bbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = (int(bbox.find(pos).text) for pos in [
                                      'xmin', 'ymin', 'xmax', 'ymax'])
            h_ratio = (ymax - ymin) / im_h
            w_ratio = (xmax - xmin) / im_w

            if params['size_ratio']:
                lr[name]['Height'].append(h_ratio)
                lr[name]['Width'].append(w_ratio)

            if params['small_object']:
                for i in sorted(params['split_range']):
                    if h_ratio < i / 100 and w_ratio < i / 100:
                        details = (  # (image1.xml, cat, w: 0.2153, h: 0.2507)
                            os.path.basename(xml), 
                            name, 
                            f'h:{h_ratio:.4f}', 
                            f'w:{w_ratio:.4f}')
                        sc[f'{i}x{i}'].append(details)
                        break

    return lc, lr, sc


def merge_counts(lc, lr, sc, new_lc, new_lr, new_sc):
    '''
    Merge the results of all XML files for label counts,
    label ratios and small-object counts.

    Args:
        lc (dict): Final result of label counts.
        lr (dict): Final result of label size ratios.
        sc (dict): Final result of small-object counts.
        new_lc (dict): New label counts for merging.
        new_lr (dict): New label ratios for merging.
        new_sc (dict): New small-object counts for merging.

    Returns:
        None.
    '''
    # Merge label counts
    for key, value in new_lc.items():
        lc[key] = lc.get(key, 0) + value

    # Merge label ratios
    for label, ratios in new_lr.items():
        if label not in lr:
            lr[label] = {'Height': [], 'Width': []}
        lr[label]['Height'].extend(ratios['Height'])
        lr[label]['Width'].extend(ratios['Width'])

    # Merge small-object counts
    for sr, count in new_sc.items():
        sc[sr].extend(count)


def count(params):
    '''
    Main process of counting labels.

    Args:
        params (dict): Parameters.

    Returns:
        lc (dict): Final result of label counts.
        lr (dict): Final result of label size ratios.
        sc (dict): Final result of small-object counts.
    '''
    lc = {}
    lr = {}
    sc = {f'{x}x{x}': [] for x in params['split_range']}

    if isinstance(params['xmls'], str):  # Directory of annotations
        xmls = os.listdir(params['xmls'])
    elif isinstance(params['xmls'], list):  # List of annotation file
        xmls = params['xmls']
    else:
        print("Unsupported input.(params['xmls'])")
        exit()

    # Process each XML file in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_xml, params, xml) for xml in xmls
        ]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Counting'):
            new_lc, new_lr, new_sc = future.result()
            merge_counts(lc, lr, sc, new_lc, new_lr, new_sc)

    lc = dict(sorted(lc.items()))

    return lc, lr, sc


def build_results_dir(params):
    '''
    Create results saving folder.

    Args:
        params (dict): Parameters.

    Returns:
        params (dict): Parameters.
    '''
    if isinstance(params['xmls'], list):
        return params

    data_dir = os.path.dirname(params['xmls'])
    ratios_dir = small_dir = None

    t = time.strftime('%Y%m%d_%H%M%S')
    # Directory to save results.
    save_dir = os.path.join(data_dir, 'Statistical_Results', t)
    os.makedirs(save_dir, exist_ok=True)

    if params['size_ratio']:
        # Directory of ratio results.
        ratios_dir = os.path.join(save_dir, 'label_size_ratios')
        os.makedirs(ratios_dir, exist_ok=True)

    if params['small_object']:
        # Directory of small-objects results.
        small_dir = os.path.join(save_dir, 'small_object_counts')
        os.makedirs(small_dir, exist_ok=True)

    params['save_dir'] = save_dir
    params['ratios_dir'] = ratios_dir
    params['small_dir'] = small_dir

    return params


def plot_label_counts(lc, params):
    '''
    Plot result of label counts.

    Args:
        lc (dict): Final label counts.
        params (dict): Parameters.

    Returns:
        params (dict): Parameters.
    '''
    # Colors used to plot
    colorset = [
        (218, 179, 218), (138, 196, 208), (112, 112, 181), (255, 160, 100), 
        (106, 161, 115), (232, 190,  93), (211, 132, 252), ( 77, 190, 238), 
        (  0, 170, 128), (196, 100, 132), (153, 153, 153), (194, 194,  99), 
        ( 74, 134, 255), (205, 110,  70), ( 93,  93, 135), (140, 160,  77), 
        (255, 185, 155), (255, 107, 112), (165, 103, 190), (202, 202, 202), 
        (  0, 114, 189), ( 85, 170, 128), ( 60, 106, 117), (250, 118, 153), 
        (119, 172,  48), (171, 229, 232), (160,  85, 100), (223, 128,  83), 
        (217, 134, 177), (133, 111, 102), 
    ]
    colorset = [(r/255, g/255, b/255) for r, g, b in colorset]
    params['colorset'] = colorset

    keys = list(lc.keys())
    values = list(lc.values())
    print(f'Dataset has {len(keys)} classes and {sum(values)} labels in total.')
    print(lc)

    if isinstance(params['xmls'], str):
        print('Plotting label counts...')
    
    # Calculate figure size based on number of keys
    fig_width = max(len(keys) * 0.4, 6)  # Adjust multiplier as needed
    plt.figure(figsize=(fig_width, 6))

    bars = plt.bar(
        keys, 
        values, 
        color=[colorset[i % len(colorset)] for i in range(len(keys))], 
        alpha=0.5)
    for x, y in zip(keys, values):
        plt.text(x, y, y, ha='center', va='bottom')

    plt.title('Label Counts')
    plt.xticks(rotation=90)
    plt.ylabel('Counts')
    plt.xlim(-0.8, len(keys))  # Delete unneeded white space
    if isinstance(params['xmls'], str):
        plt.savefig(
            os.path.join(params['save_dir'], 'label_counts.jpg'),
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=200)
    plt.close()

    return params


def plot_label_ratios(lr, params):
    '''
    Plot result of label ratios.

    Args:
        lr (dict): Final label size ratios.
        params (dict): Parameters.

    Returns:
        None.
    '''
    print('Plotting label ratios...')
    color = {'Height': None, 'Width': 'red'}
    for label, ratios in lr.items():
        # Plot histogram for height / width ratios
        for key in ratios.keys():  # key: Height, Width
            plt.hist(
                ratios[key],
                bins=20,
                alpha=0.5,
                color=color[key],
                label='Ratio Counts')
            plt.title(f'{label} {key} Ratios')
            plt.xlabel('Ratio')
            plt.ylabel('Counts')
            plt.legend()
            if isinstance(params['xmls'], str):
                plt.savefig(
                    os.path.join(params['ratios_dir'], f'{label}_{key}_Ratios.jpg'),
                    bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=200)
            plt.close()


def plot_small_counts(sc, params):
    '''
    Plot a stacked bar chart of small-object counts by label and
    save detailed info as txt.

    Args:
        sc (dict): Final result of small-object counts.
        params (dict): Parameters.

    Returns:
        None.
    '''
    print('Plotting small-object counts by label...')

    # Get unique labels for legend of small objects counting chart
    unique_labels = set()
    for size in sc.values():
        for item in size:
            unique_labels.add(item[1])

    label_to_color = {label: params['colorset'][i % len(params['colorset'])] 
                      for i, label in enumerate(unique_labels)}

    # Prepare data for each size range
    plot_data = {f'{x}x{x}': {label: 0 for label in unique_labels}
                 for x in params['split_range']}
    for size, items in sc.items():
        for item in items:
            label = item[1]
            plot_data[size][label] += 1

    x_labels = [f'< {x}% x {x}%' for x in params['split_range']]

    # Draw stacked bar chart
    plt.figure(figsize=(10, 6))
    sum_per_cat = np.zeros(len(params['split_range']))

    for label, color in label_to_color.items():
        counts = [plot_data[f'{x}x{x}'][label]
                  for x in params['split_range']]
        plt.bar(
            x_labels, 
            counts, 
            bottom=sum_per_cat, 
            label=label, 
            color=color)
        sum_per_cat += np.array(counts)

    # Make the categories in the bar and the legend has same order.
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.reverse()
    labels.reverse()

    # Add total count above each bar
    for i, total in enumerate(sum_per_cat):
        plt.text(i, total, int(total), ha='center', va='bottom')

    plt.title('Small Object Counts by Size Range and Label')
    plt.xticks(range(len(params['split_range'])), x_labels)
    plt.xlabel('Size Range')
    plt.ylabel('Counts')
    plt.legend(
        handles, 
        labels, 
        title='Labels', 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left')
    plt.tight_layout()
    if isinstance(params['xmls'], str):
        plt.savefig(os.path.join(
            params['small_dir'], 
            'small_object_counts_by_label.jpg'))
    plt.close()

    # Save detailed info
    for x in params['split_range']:
        filename = os.path.join(params['small_dir'], f'smaller_than_{x}%x{x}%.txt')
        with open(filename, 'w') as f:
            for i in sc[f'{x}x{x}']:
                f.write(' '.join(map(str, i)) + '\n')


def run(params):
    '''
    Main process.

    Args:
        params (dict): Parameters.

    Returns:
        None.
    '''
    lc, lr, sc = count(params)
    params = build_results_dir(params)
    params = plot_label_counts(lc, params)
    if params['size_ratio']:
        plot_label_ratios(lr, params)
    if params['small_object']:
        plot_small_counts(sc, params)
    print('Counting completed.')
    if isinstance(params['xmls'], str):
        print(f'Counting result(s) saved at {params["save_dir"]}.')


if __name__ == '__main__':
    # Setting parameters
    params = {
        'xmls': 'path/of/xml/files',  # Directory of annotations or list of absolute paths of annotations.
        'size_ratio': True,  # Weather to count size ratio per class.
        'small_object': True,  # Weather to count small objects.
        'split_range': [1, 2, 3, 4, 5]  # Count label smaller than 1% × 1%, etc.
    }

    run(params)
