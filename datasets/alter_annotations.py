"""
This script performs a requested alteration onto a set of annotations images.
"""

import argparse
import multiprocessing
import shutil
from pathlib import Path
from pprint import pprint
from colorama import Fore, Style, init

from joblib import Parallel, delayed
from tqdm import tqdm

from PIL import Image
import numpy as np

from utils.files import retrieve_dataset_information
from utils.processing import dilate_image, erode_image

import matplotlib.pyplot as plt

from utils.validation import SegmentationEvaluator

AVAILABLE_TRANSFORMATIONS = {
    'dilation': dilate_image,
    'erosion': erode_image
}


def process_annotation(annotation_path, class_names, class_colors_dictionary, initial_size,
                       target_average_miou,
                       precision_tolerance,
                       transformation,
                       actual_target_path,
                       merge_classes):
    MAX_SIZE = 100
    annotation_name = annotation_path.stem

    image = Image.open(annotation_path)
    image.load()
    data = np.asarray(image, dtype=np.uint8)
    data = data.astype(np.uint8)
    result = None

    classes_similarities = {}

    for class_index in range(len(class_names)):
        class_name = class_names[class_index]

        red, green, blue = class_colors_dictionary[class_name]
        class_indices = np.where((data[:, :, 0] == red) & (data[:, :, 1] == green) & (data[:, :, 2] == blue))

        is_class_present = class_indices[0].size != 0 or class_indices[1].size != 0

        if is_class_present:
            if result is None:
                result = np.zeros_like(data)
                result = result.astype('uint8')

            # Apply transformation with size that fits best.
            current_size = initial_size
            current_miou = 1.0
            transformed_data = data

            while current_size < MAX_SIZE and abs(current_miou - target_average_miou) > precision_tolerance:
                # print('Class {}, current mIOU: {}, target: {}, current size: {}'.format(
                #    class_name, current_miou, target_average_miou, current_size))
                transformed_data = AVAILABLE_TRANSFORMATIONS[transformation](binary_mask=data, size=current_size)
                current_miou = SegmentationEvaluator.compute_mean_iou(transformed_data, data)
                current_size += 1

            # print('Class {} has mIOU {} with size {}'.format(class_name, current_miou, current_size))
            classes_similarities[class_name] = current_miou

            # Generate the colored mask based on Pascal's palette.
            colored_mask = np.zeros_like(transformed_data)
            class_indices = np.where((transformed_data[:, :, 0] == red)
                                     & (transformed_data[:, :, 1] == green)
                                     & (transformed_data[:, :, 2] == blue))
            colored_mask[class_indices] = 1

            # Colorize the binary mask with the class color.
            colored_mask[:, :, 0] *= red
            colored_mask[:, :, 1] *= green
            colored_mask[:, :, 2] *= blue

            if merge_classes:
                result[class_indices] = colored_mask[class_indices]
            else:
                image = Image.fromarray(colored_mask)
                image.save(Path(actual_target_path, class_name, annotation_name + '.png'))

    if merge_classes and result is not None:
        result = Image.fromarray(result)
        result.save(Path(actual_target_path, annotation_name + '.png'))

    image_similarity = sum(classes_similarities.values()) / len(classes_similarities.keys())
    print(Fore.GREEN + 'Processed ' + Style.RESET_ALL + annotation_name + ' : ' + Fore.CYAN + ' ' +
          '{:04.2f}'.format(image_similarity * 100) + ' %')

    return image_similarity


def main(dataset_root_path, annotation_images_path, actual_target_path, transformation, initial_size,
         target_average_miou, precision_tolerance, merge_classes):
    if transformation not in AVAILABLE_TRANSFORMATIONS.keys():
        raise ValueError('Requested transformation {} is not supported. '
                         'Must be one of {}.'.format(transformation, list(AVAILABLE_TRANSFORMATIONS.keys())))

    class_names, class_colors = retrieve_dataset_information(dataset_path=dataset_root_path)
    class_colors_dictionary = dict(zip(class_names, class_colors))

    actual_target_path = Path(actual_target_path, '{}_miou_{:04.2f}_pm_{:04.2f}'.format(transformation,
                                                                                        target_average_miou * 100,
                                                                                        precision_tolerance * 100))
    actual_target_path.mkdir(exist_ok=True)
    # Print the color palette retrieved from the dictionary at the root of the dataset directory.
    for class_name in class_names:
        r, g, b = class_colors_dictionary[class_name]

        class_directory = Path(actual_target_path, class_name)
        if not merge_classes:
            class_directory.mkdir(parents=True, exist_ok=True)

        # print(class_name, r, g, b, sep=',')

    # Transform every annotation image.
    images_similarities = Parallel(n_jobs=5)(delayed(process_annotation)
                                             (annotation_path,
                                              class_names,
                                              class_colors_dictionary,
                                              initial_size,
                                              target_average_miou,
                                              precision_tolerance,
                                              transformation,
                                              actual_target_path,
                                              merge_classes)
                                             for annotation_path
                                             in Path(annotation_images_path).glob('*.png'))
    average_similarity = sum(images_similarities) / len(images_similarities)

    # with open(Path(actual_target_path, 'stats.txt'), 'wt') as out:
    #    pprint(global_class_similarities, stream=out)

    print('Average similarity for transformed set: {:4.2f} %'.format(average_similarity * 100))
    new_directory_name = actual_target_path.with_name('{}_{:04.2f}_{:04.2f}'.format(transformation,
                                                                                    target_average_miou * 100,
                                                                                    average_similarity * 100))
    print('Renaming {} in {}'.format(actual_target_path, new_directory_name))
    shutil.move(str(actual_target_path), str(new_directory_name))


if __name__ == "__main__":
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
    parser.add_argument('--merge-classes', action='store', default=True, type=bool,
                        help='Whether or not to merge the different masks into a single annotation.')
    parser.add_argument('--transformation', action='store',
                        help='The kind of transformation to apply to the class binary masks.')
    parser.add_argument('--initial-size', action='store',
                        default=0,
                        help='The initial size of the kernel to use during transformations.')
    parser.add_argument('--target-average-miou', action='store', type=float,
                        help='''The average mIOU that the resulting modified dataset should have.
                             This is leveraged by the precision tolerance.''')
    parser.add_argument('--precision-tolerance', action='store', type=float,
                        help='The tolerance around the target precision.')
    args = parser.parse_args()

    DATASET_ROOT_PATH = Path(str(args.dataset_root_path))
    ANNOTATION_IMAGES_PATH = Path(str(args.annotations_path))
    TARGET_PATH = Path(str(args.target_path))
    TRANSFORMATION = str(args.transformation)
    INITIAL_SIZE = int(args.initial_size)
    TARGET_AVERAGE_MIOU = float(args.target_average_miou)
    PRECISION_TOLERANCE = float(args.precision_tolerance)
    MERGE_CLASSES = bool(args.merge_classes)

    init()

    print("Using " + Fore.BLUE + str(5) + Style.RESET_ALL + " CPUs.")

    main(DATASET_ROOT_PATH, ANNOTATION_IMAGES_PATH, TARGET_PATH, TRANSFORMATION,
         INITIAL_SIZE, TARGET_AVERAGE_MIOU, PRECISION_TOLERANCE, MERGE_CLASSES)
