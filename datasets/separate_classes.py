"""
This modules separates image annotations into separate annotations for each class. These can then be used by the
convert_to_rgb_palette.py script to turn the dataset into the form accepted by the RevaNet framework.

If you're not trying to perform multi-label classification, you're at the wrong place.
"""

import argparse
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
                    help='Where to look for the annotation images.')
parser.add_argument('--target-path', dest='target_path', action='store',
                    help='The path where the transformed annotations should be saved. '
                         'Where to store the temporary, separated individual class masks.')
args = parser.parse_args()

DATASET_ROOT_PATH = Path(str(args.dataset_root_path))
ANNOTATION_PATH = Path(str(args.annotations_path))
TARGET_PATH = Path(str(args.target_path))

class_names, class_colors = retrieve_dataset_information(dataset_path=DATASET_ROOT_PATH)
class_colors_dictionary = dict(zip(class_names, class_colors))

print(ANNOTATION_PATH)
print(TARGET_PATH)

# Print the color palette retrieved from the dictionary at the root of the dataset directory.
for class_name in class_names:
    r, g, b = class_colors_dictionary[class_name]
    class_directory = Path(TARGET_PATH, class_name)
    class_directory.mkdir(parents=True, exist_ok=True)

    print(class_name, r, g, b, sep=',')

# Transform every annotation image.

for annotation_path in tqdm(Path(ANNOTATION_PATH).glob('*.png')):
    annotation_name = annotation_path.stem
    target_path = TARGET_PATH

    image = Image.open(annotation_path)
    image.load()
    data = np.asarray(image, dtype=np.uint8)
    data = data.astype(np.uint8)

    for class_index in range(len(class_names)):
        class_name = class_names[class_index]

        red, green, blue = class_colors_dictionary[class_name]
        class_indices = np.where((data[:, :, 0] == red) & (data[:, :, 1] == green) & (data[:, :, 2] == blue))

        is_class_present = class_indices[0].size != 0 or class_indices[1].size != 0

        if is_class_present:
            # Generate the colored mask based on Pascal's palette.
            colored_mask = np.zeros_like(data)
            colored_mask[class_indices] = 1
            colored_mask[:, :, 0] *= red
            colored_mask[:, :, 1] *= green
            colored_mask[:, :, 2] *= blue

            image = Image.fromarray(colored_mask)
            image.save(Path(target_path, class_name, annotation_name + '.png'))
