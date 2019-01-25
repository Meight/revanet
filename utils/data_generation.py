import random
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

import imgaug as ia
import imgaug.augmenters as augmenters
from utils.augmentation import resize_to_size
from utils.files import retrieve_dataset_information
from utils.segmentation import image_to_one_hot, one_hot_to_image
from utils.utils import load_image, prepare_data


class DataGenerator():
    def __init__(self, number_of_epochs, batch_size, subset_associations,
                 class_colors, input_size):
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.current_epoch = 0
        self.current_step = 0
        self.number_of_samples = len(subset_associations.keys())
        self.steps_per_epoch = self.number_of_samples // batch_size
        print(
            'There are {} samples, so {} steps per epoch with batch size {}.'.
            format(self.number_of_samples, self.steps_per_epoch,
                   self.batch_size))
        self.image_paths = list(subset_associations.keys())
        self.subset_associations = subset_associations
        self.class_colors = class_colors
        self.number_of_classes = len(class_colors)

    def _shuffle_paths(self):
        self.image_paths = np.random.permutation(self.image_paths)

    def get_batch(self):
        while self.current_epoch < self.number_of_epochs:

            images_batch = []
            annotations_batch = []

            for batch_index in range(self.batch_size):
                sample_index = (self.current_step * self.batch_size +
                                batch_index) % self.number_of_samples
                sample_path = self.image_paths[sample_index]
                image = load_image(sample_path)

                annotation_path = random.choice(
                    self.subset_associations[sample_path])
                annotation = load_image(annotation_path)

                image, annotation = resize_to_size(image, annotation,
                                                   self.input_size)
                image = np.float32(image) / 255.0

                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(annotation)
                annotation = one_hot_to_image(
                    image_to_one_hot(annotation, self.class_colors))
                # plt.title('Before one hot')
                # plt.subplot(1, 3, 2)
                # plt.imshow(annotation)
                # plt.title('After one hot')

                annotation = ia.SegmentationMapOnImage(
                    annotation,
                    shape=image.shape,
                    nb_classes=self.number_of_classes)

                # plt.subplot(1, 3, 3)
                # plt.imshow(annotation.draw)
                # plt.title('After map on image')
                # plt.savefig("./tmp/colors/" + self.current_step + ".png")

                images_batch.append(image)
                annotations_batch.append(annotation)

            batch = ia.Batch(
                images=np.array(images_batch, dtype=np.float32),
                segmentation_maps=annotations_batch)

            self.current_step += 1

            self._check_epoch_end()

            yield batch

    def _check_epoch_end(self):
        if self.current_step == self.steps_per_epoch:
            self.current_epoch += 1
            self.current_step = 0
            self._shuffle_paths()

    def __repr__(self):
        return '''Data generator over {} epochs, with {} samples and
        a batch size of {} (meaning {} steps per epoch).'''.format(
            self.number_of_epochs, self.number_of_samples, self.batch_size,
            self.steps_per_epoch)


def get_batch_loader_for_subset(number_of_epochs,
                                batch_size,
                                input_size,
                                subset_associations,
                                class_colors,
                                strategy='none'):

    base_augmenters = augmenters.SomeOf(3, [
        augmenters.Grayscale(alpha=(0.0, 1.0)),
        augmenters.ContrastNormalization((0.5, 1.5)),
        augmenters.Affine(scale=(0.5, 1.5)),
        augmenters.Affine(rotate=(-20, 20)),
        augmenters.CropAndPad(percent=(-0.15, 0.15)),
        augmenters.Sharpen((0.0, 1.0)),
    ])

    aggressive_augmenters = augmenters.append(
        augmenters.OneOf([
            augmenters.Fog(deterministic=True),
            augmenters.Snowflakes(deterministic=True),
            augmenters.FastSnowyLandscape(deterministic=True),
            augmenters.GaussianBlur(sigma=(0.0, 2.0), deterministic=True),
            augmenters.Add((-20, 20), per_channel=0.5, deterministic=True),
            augmenters.CoarseDropout(
                0.02, size_percent=0.05, per_channel=0.5, deterministic=True),
            augmenters.AdditiveGaussianNoise(
                scale=0.1 * 255, deterministic=True),
            augmenters.AddElementwise((-10, 10),
                                      per_channel=0.5,
                                      deterministic=True),
            augmenters.Emboss(
                alpha=(0.0, 0.5), strength=(0.5, 1.5), deterministic=True)
        ]))

    if strategy == 'light':
        used_augmenters = [base_augmenters]
    elif strategy == 'aggresive':
        used_augmenters = [aggressive_augmenters, base_augmenters]
    else:
        used_augmenters = []

    augmentation_pipeline = augmenters.Sequential(
        used_augmenters, deterministic=True)

    data_generator = DataGenerator(number_of_epochs, batch_size,
                                   subset_associations, class_colors,
                                   input_size)
    batch_loader = ia.BatchLoader(data_generator.get_batch)
    bg_augmenter = ia.BackgroundAugmenter(batch_loader, augmentation_pipeline)

    # Both get returned because they must be terminated manually.
    return bg_augmenter, batch_loader, data_generator.steps_per_epoch


if __name__ == "__main__":
    DATASET_PATH = Path('datasets/test_augmentation')
    TRAIN_PATH = Path(DATASET_PATH, 'train')
    TRAIN_ANNOTATIONS_PATH = Path(DATASET_PATH, 'train_annotations')

    subset_associations = prepare_data(TRAIN_PATH, TRAIN_ANNOTATIONS_PATH,
                                       None, None)

    class_names_list, class_colors = retrieve_dataset_information(
        dataset_path=DATASET_PATH)
    class_colors_dictionary = dict(zip(class_names_list, class_colors))
    number_of_classes = len(class_colors)

    data_generator = DataGenerator(3, 1, subset_associations['train'],
                                   class_colors)
    batch_loader = ia.BatchLoader(data_generator.get_batch)
    bg_augmenter = ia.BackgroundAugmenter(batch_loader, augmentation_pipeline)

    print(data_generator)

    i = 1
    while True:
        batch = bg_augmenter.get_batch()

        if batch is None:
            print('Went through dataset.')
            break

        augmented_images = batch.images_aug
        augmented_annotations = batch.segmentation_maps_aug

        imageio.imwrite('test{}.jpg'.format(i), augmented_images[0])
        i += 1

    batch_loader.terminate()
    bg_augmenter.terminate()
