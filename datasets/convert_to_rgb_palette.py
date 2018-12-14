"""
This module helps converting any dataset with grayscale-based annotations to their RGB equivalent.
It uses the palette provided in the CSV dictionary at the root of the dataset.
"""

import argparse
import os
import sys
from pathlib import Path

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
parser.add_argument('--is-for-multi-label', dest='is_for_multi_label', action='store_true',
                    default=False,
                    help='Whether or not to generate subfolders for each class in the subsets.')
args = parser.parse_args()

DATASET_ROOT_PATH = Path(str(args.dataset_root_path))
ANNOTATION_PATH = Path(str(args.annotations_path))
TARGET_PATH = Path(str(args.target_path))
IS_FOR_MULTI_LABEL = bool(args.is_for_multi_label)

class_names, class_colors = retrieve_dataset_information(dataset_path=DATASET_ROOT_PATH)
class_colors_dictionary = dict(zip(class_names, class_colors))

print(ANNOTATION_PATH)
print(TARGET_PATH)
print('Multi-label dataset generation:', IS_FOR_MULTI_LABEL)

if IS_FOR_MULTI_LABEL:
    for class_index in class_names:
        class_origin_path = Path(ANNOTATION_PATH, class_index)
        if not os.path.exists(class_origin_path):
            raise ValueError('Class {} (given in the CSV configuration file) was not found at the provided '
                             'annotation_path ({}). Did you make a typo?'.format(class_index, ANNOTATION_PATH))

        class_full_path = Path(TARGET_PATH, class_index)
        if not os.path.exists(class_full_path):
            class_full_path.mkdir(parents=True, exist_ok=True)

# Print the color palette retrieved from the dictionary at the root of the dataset directory.
for class_index in class_names:
    r, g, b = class_colors_dictionary[class_index]
    print(class_index, r, g, b, sep=',')

TARGET_PATH.mkdir(parents=True, exist_ok=True)

# Transform every annotation image.

for annotation_path in tqdm(Path(ANNOTATION_PATH).glob('**/*.png' if IS_FOR_MULTI_LABEL else '*.png')):
    annotation_name = annotation_path.stem
    target_path = TARGET_PATH

    if IS_FOR_MULTI_LABEL:
        # Retrieve the current class.
        target_path = Path(TARGET_PATH, annotation_path.parent.name)

    image = Image.open(annotation_path)
    image.load()
    data = np.asarray(image, dtype=np.uint8)

    result = None

    if IS_FOR_MULTI_LABEL:
        result = np.ones_like(data)
        result = result.astype('uint8')
        result = np.stack((result,) * 3, axis=-1)
        background_red, background_green, background_blue = class_colors_dictionary['background']
        result[:, :, 0] *= background_red
        result[:, :, 1] *= background_green
        result[:, :, 2] *= background_blue

        class_name = annotation_path.parent.name
        class_indices = np.where(data.astype(np.uint8) == 255)

        # Generate the colored mask based on Pascal's palette.
        colored_mask = np.zeros_like(data)
        colored_mask[class_indices] = 1
        colored_mask = np.stack((colored_mask,) * 3, axis=-1)
        red, green, blue = class_colors_dictionary[class_name]
        colored_mask[:, :, 0] *= red
        colored_mask[:, :, 1] *= green
        colored_mask[:, :, 2] *= blue

        result[class_indices] = colored_mask[class_indices]
    else:
        for class_index in range(len(class_names)):
            class_name = class_names[class_index]
            if result is None:
                result = np.zeros_like(data)
                result = result.astype('uint8')
                result = np.stack((result,) * 3, axis=-1)

            class_indices = np.where(data.astype(np.uint8) == class_index)

            # Generate the colored mask based on Pascal's palette.
            colored_mask = np.zeros_like(data)
            colored_mask[class_indices] = 1
            colored_mask = np.stack((colored_mask,) * 3, axis=-1)
            red, green, blue = class_colors_dictionary[class_name]
            colored_mask[:, :, 0] *= red
            colored_mask[:, :, 1] *= green
            colored_mask[:, :, 2] *= blue

            result[class_indices] = colored_mask[class_indices]

    image = Image.fromarray(result)
    image.save(Path(target_path, annotation_name + '.png'))
    # print('Saved', annotation_name, 'to', Path(target_path, annotation_name + '.png'))
    # sys.stdout.flush()
