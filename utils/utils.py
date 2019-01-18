from __future__ import division, print_function

import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

import cv2
import tensorflow as tf
from utils.augmentation import resize_to_size


def gather_multi_label_data(dataset_directory):
    """
    Retrieves all images per subset for the training, validation and testing phases.

    For each image, retrieves the associated masks for each class present within the dataset. There may be several
    of such annotations for any given class. The function expects these annotations to have their name beginning
    with the their associated image's name.

    'train':
        'image1.png':
            'background':
                ['mask1', 'mask2']
            'person':
                ['mask1']

    :param dataset_directory: the name of the directory where to look for the subsets.
    :return:
    """
    paths = {}

    for subset_name in ['train', 'val', 'test']:
        paths[subset_name] = {}
        subset_annotations_path = subset_name + '_labels'
        cwd = os.getcwd()

        class_directories = os.listdir(os.path.join(cwd, dataset_directory, subset_annotations_path))

        for image_name in os.listdir(os.path.join(cwd, dataset_directory, subset_name)):
            image_path = os.path.join(cwd, dataset_directory, subset_name, image_name)
            image_masks = {}

            for current_class_directory in class_directories:
                current_class_masks = glob.glob(os.path.join(cwd,
                                                             dataset_directory,
                                                             subset_annotations_path,
                                                             current_class_directory,
                                                             os.path.splitext(image_name)[0] + '*'))

                if current_class_masks:
                    image_masks[current_class_directory] = current_class_masks

            paths[subset_name][image_path] = image_masks

    return paths


def get_available_annotation_resized_tensors_for_image(input_shape,
                                                       image_masks_dictionary,
                                                       class_colors_dictionary,
                                                       mode='linear'):
    available_modes = ['linear']
    n_hot_encoded_tensors = []

    if not image_masks_dictionary.values():
        return n_hot_encoded_tensors

    if not mode in available_modes:
        raise Exception('Provided mode {} is not supported for tensors generation in multi-label classification.'.
                        format(mode))

    if mode == 'linear':
        # In linear mode, all classes are expected to have the same amount of masks so that one tensor can
        # be created "vertically."
        masks_count = len(next(iter(image_masks_dictionary.values())))
        blank_mask = np.ones(input_shape)

        for k in range(masks_count):
            different_classes_one_hot = []
            for class_name, class_colors in class_colors_dictionary.items():
                class_mask = blank_mask

                if class_name in image_masks_dictionary.keys():
                    mask_image, _ = resize_to_size(load_image(image_masks_dictionary[class_name][k]),
                                                   desired_size=input_shape[0])
                    equality = np.equal(mask_image, class_colors)
                    class_mask = np.all(equality, axis=-1)

                print('Mask shape', np.shape(class_mask))
                different_classes_one_hot.append(class_mask)

            if different_classes_one_hot:
                n_hot_encoded_tensors.append(np.stack(different_classes_one_hot, axis=-1))

    return n_hot_encoded_tensors


def to_n_hot_encoded(masks_dictionary, class_names):
    return np.asarray([1 if class_name in masks_dictionary.keys() else 0 for class_name in class_names])


def prepare_data(train_path: Path,
                 train_annotations_path: Path,
                 validation_path: Path,
                 validation_annotations_path: Path):
    paths_associations = {
        'train': {
            'images_path': train_path,
            'annotations_path': train_annotations_path
        },
        'validation': {
            'images_path': validation_path,
            'annotations_path': validation_annotations_path
        }
    }

    associations = {}

    for split_name, split_data in paths_associations.items():
        if split_data['images_path'] is None or split_data['annotations_path'] is None:
            continue

        # Got to use an ordered dictionary to ensure reproducibility, as order isn't guaranteed otherwise and we'll be
        # picking samples randomly in this set.
        associations[split_name] = OrderedDict()

        for image_file_name in split_data['images_path'].glob('*'):
            image_name = image_file_name.stem
            associations[split_name][image_file_name.absolute()] = [annotation_path.absolute()
                                                                    for annotation_path
                                                                    in split_data['annotations_path'].glob(image_name
                                                                                                           + '*')]

    return associations


def prepare_data_bak(dataset_directory):
    train_input_names = []
    train_output_names = []
    val_input_names = []
    val_output_names = []
    test_input_names = []
    test_output_names = []
    for file in os.listdir(dataset_directory + "/train"):
        cwd = os.getcwd()
        train_input_names += glob.glob(cwd + "/" + dataset_directory + "/train/" + file + "*")
    for file in os.listdir(dataset_directory + "/train_labels"):
        cwd = os.getcwd()
        train_output_names += glob.glob(cwd + "/" + dataset_directory + "/train_labels/" + file + "*")
    for file in os.listdir(dataset_directory + "/val"):
        cwd = os.getcwd()
        val_input_names += glob.glob(cwd + "/" + dataset_directory + "/val/" + file + "*")
    for file in os.listdir(dataset_directory + "/val_labels"):
        cwd = os.getcwd()
        val_output_names += glob.glob(cwd + "/" + dataset_directory + "/val_labels/" + file + "*")
    for file in os.listdir(dataset_directory + "/test"):
        cwd = os.getcwd()
        test_input_names += glob.glob(cwd + "/" + dataset_directory + "/test/" + file + "*")
    for file in os.listdir(dataset_directory + "/test_labels"):
        cwd = os.getcwd()
        test_output_names += glob.glob(cwd + "/" + dataset_directory + "/test_labels/" + file + "*")
    train_input_names.sort(), train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names


# Takes an absolute file path and returns the name of the file without th extension
def file_path_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name


def count_parameters():
    total_parameters = 0

    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    return total_parameters


# Subtracts the mean images from ImageNet
def substract_image_mean(inputs, means=[123.68, 116.78, 103.94]):
    inputs = tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def filter_valid_entries(prediction, label):
    valid_indices = np.where(label != 255)

    return label[valid_indices], prediction[valid_indices]


def save_image(npdata, out_filename):
    image = Image.fromarray(npdata)
    image.save(out_filename)


def load_image(path):
    path = str(path)
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    return image


def build_images_association_dictionary(input_image_names, output_image_names):
    association_dictionary = {}

    for input_image_name in input_image_names:
        association_dictionary[input_image_name] = [image_name
                                                    for image_name in output_image_names
                                                    if os.path.splitext(os.path.basename(input_image_name))[0]
                                                    in image_name]

    return association_dictionary
