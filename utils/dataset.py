"""
Set of utility functions to ensure the correctness of a dataset.
"""

import random
from pathlib import Path

import numpy as np

import tensorflow as tf
from utils import utils
from utils.segmentation import colour_code_segmentation, image_to_one_hot


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class BaseValidationError(ValueError):
    pass


class DatasetDoesNotExistError(BaseValidationError):
    pass


class DatasetSubsetDoesNotExistError(BaseValidationError):
    pass


def check_dataset_correctness(dataset_name,
                              dataset_path: Path,
                              train_path: Path,
                              train_annotations_path: Path,
                              validation_path: Path,
                              validation_annotations_path: Path):
    """
    Checks the correctness of a provided dataset's structure.

    :param validation_annotations_path:
    :param train_annotations_path:
    :param dataset_name:    The name of the dataset to check.
    :param dataset_path:    The path to the dataset root folder.
    :param train_path:      The name of the folder containing the training samples.
    :param validation_path: The name of the folder containing the validation samples.
    """
    if not dataset_path.exists():
        raise DatasetDoesNotExistError(
            'No directory was found at {} for dataset {}.'
            .format(dataset_path,
                    dataset_name))

    if not train_path.exists():
        raise DatasetSubsetDoesNotExistError(
            'Train folder not found at {} for dataset {}.'
            .format(train_path,
                    dataset_name))

    if not train_annotations_path.exists():
        raise DatasetSubsetDoesNotExistError(
            '''No annotations folder found for train
            subset at {} for dataset {}.'''
            .format(train_annotations_path, dataset_name))

    if not validation_path.exists():
        raise DatasetSubsetDoesNotExistError(
            '''Validation folder not found
            at {} for dataset {}.'''
            .format(validation_path,
                    dataset_name))

    if not validation_annotations_path.exists():
        raise DatasetSubsetDoesNotExistError(
            '''No annotations folder found for
            validation subset at {} for dataset {}.'''
            .format(validation_annotations_path, dataset_name))


class AugmenterProxy:
    """
    This proxy must be because the augmenter's interface is kind of tricky
    and uselessly (in our case) complex. It simply proxies the generation
    to call the right methods based on that interface.

    This thing was terribly documented is a bit of a work around.
    """
    def __init__(self, background_augmenter, class_colors):
        self.background_augmenter = background_augmenter
        self.class_colors = class_colors

    def __call__(self):
        while True:
            batch = self.background_augmenter.get_batch()

            if batch is None:
                break

            # Augmented images here are not yet normalized!
            yield batch.images_aug[0], batch.segmentation_maps_aug[0].arr


def generate_dataset(background_augmenter, input_size, number_of_epochs,
                     batch_size, number_of_cpus=4, number_of_gpus=1, ratio=1, class_colors=None):
    generator = AugmenterProxy(background_augmenter, class_colors)
    parser = ExampleParser(input_size=input_size)

    images_shape = (input_size, input_size, 3)
    annotations_shape = (input_size, input_size, len(class_colors))

    training_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32), output_shapes=(images_shape, annotations_shape))
    training_dataset = training_dataset.shuffle(buffer_size=3)
    #training_dataset = training_dataset.map(
    #    parser, num_parallel_calls=number_of_cpus)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.repeat(number_of_epochs)
    training_dataset = training_dataset.prefetch(number_of_gpus)

    return training_dataset


def serialize_example(image_path, annotation_path):
    example_image = np.array(utils.load_image(image_path))
    example_annotation = np.array(utils.load_image(annotation_path))

    height = example_image.shape[0]
    width = example_image.shape[1]

    image_row = example_image.tostring()
    annotation_raw = example_annotation.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(image_row),
        'mask_raw': _bytes_feature(annotation_raw)
    }))

    return example.SerializeToString()


def load_example(item):
    key, value = item

    return serialize_example(str(key), str(random.choice(value)))


class ExampleParser:
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, serialized_example, *args, **kwargs):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        example_image = tf.decode_raw(features['image_raw'], tf.uint8)
        example_annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        annotation_shape = tf.stack([height, width, 3])

        example_image = tf.reshape(example_image, image_shape)
        example_annotation = tf.reshape(example_annotation, annotation_shape)

        annotation_shifted_classes = tf.cast(example_annotation, tf.int32) + 1

        cropped_padded_img = tf.image.resize_image_with_pad(
            example_image, self.input_size, self.input_size)
        cropped_padded_annotation = tf.image.resize_image_with_pad(
            annotation_shifted_classes, self.input_size, self.input_size)

        annotation_additional_mask_out = tf.to_int32(
            tf.equal(cropped_padded_annotation, 0)) * (255 + 1)

        cropped_padded_annotation = tf.cast(
            cropped_padded_annotation, tf.int32) + annotation_additional_mask_out - 1
        example_image = tf.cast(cropped_padded_img, tf.float32) / 255.0
        # example_annotation = tf.cast(segmentation.image_to_one_hot(annotation=example_annotation,
        #                                                           class_colors=class_colors), tf.float32)

        return example_image, cropped_padded_annotation
