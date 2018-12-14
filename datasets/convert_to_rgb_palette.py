"""
This module helps converting any dataset with grayscale-based annotations to their RGB equivalent.
It uses the palette provided in the CSV dictionary at the root of the dataset.
"""

import argparse
import glob
import os
import sys

from tqdm import tqdm

from PIL import Image
import numpy as np

from utils.files import retrieve_dataset_information

parser = argparse.ArgumentParser(description='Turns a given dataset into TF records.')
parser.add_argument('--dataset-root-path', dest='dataset_root_path', action='store',
                    default='/projets/thesepizenberg/deep-learning/datasets/VOC2012-fresh',
                    help='Name of the method to generate the TF records with.')
parser.add_argument('--annotations-path', dest='annotations_path', action='store',
                    default='/projets/thesepizenberg/deep-learning/datasets/VOC2012-fresh/SegmentationClass',
                    help='Name of the method to generate the TF records with.')
parser.add_argument('--target-path', dest='target_path', action='store',
                    help='The path where the transformed annotations should be saved. '
                         'This is usually the train_label directory of your dataset.')
args = parser.parse_args()

DATASET_ROOT_PATH = str(args.dataset_root_path)
ANNOTATION_PATH = str(args.annotations_path)
TARGET_PATH = str(args.target_path)

class_names, class_colors = retrieve_dataset_information(dataset_path=DATASET_ROOT_PATH)
class_colors_dictionary = dict(zip(class_names, class_colors))

# Print the color palette retrieved from the dictionary at the root of the dataset directory.
for class_name in class_names:
    r, g, b = class_colors_dictionary[class_name]
    print(class_name, r, g, b, sep=',')

    os.makedirs(TARGET_PATH)

# Transform every annotation image.
for annotation_path in tqdm(glob.glob(os.path.join(ANNOTATION_PATH, '*.png'))):
    annotation_name = os.path.basename(annotation_path)

    image = Image.open(annotation_path)
    image.load()
    data = np.asarray(image, dtype=np.uint8)


    result = None
    for class_name in class_names:
        if result is None:
            result = np.zeros_like(data)
            result = result.astype('uint8')
            result = np.stack((result,) * 3, axis=-1)

        class_indices = np.where(data.astype(np.uint8) == class_name)

        # Generate the colored mask based on Pascal's palette.
        colored_mask = np.zeros_like(data)
        colored_mask[class_indices] = 1
        colored_mask = np.stack((colored_mask,) * 3, axis=-1)
        red, green, blue = class_colors[class_name]
        colored_mask[:, :, 0] *= red
        colored_mask[:, :, 1] *= green
        colored_mask[:, :, 2] *= blue

        result[class_indices] = colored_mask[class_indices]

    image = Image.fromarray(result)
    image.save(os.path.join(TARGET_PATH, annotation_name))
    print('Saved', annotation_name, 'to', TARGET_PATH)
    sys.stdout.flush()

