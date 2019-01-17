import imgaug as ia
import imgaug.augmenters as augmenters
import random
import numpy as np
from utils.utils import load_image
from utils.segmentation import one_hot_to_image, image_to_one_hot


class DataGenerator():
    def __init__(self, number_of_epochs, batch_size, image_paths, actual_paths, subset_associations, class_colors):
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.current_epoch = 0
        self.current_step = 0
        self.steps_per_epoch = len(image_paths) // batch_size
        self.number_of_samples = len(image_paths)
        self.actual_paths = actual_paths
        self.subset_associations = subset_associations
        self.class_colors = class_colors

    def _shuffle_paths(self):
        self.image_paths = np.random.permutation(self.image_paths)

    def get_batch(self):
        if self.current_epoch == self.number_of_epochs:
            yield None

        images_batch = []
        annotations_batch = []

        for batch_index in range(self.batch_size):
            sample_index = (self.current_step * self.batch_size *
                            batch_index) % self.number_of_samples
            sample_index = self.image_paths[sample_index]
            sample_path = self.actual_paths[sample_index]
            image = load_image(sample_path)
            images_batch.append(image)

            annotation_path = random.choice(
                self.subset_associations[sample_path])
            annotation = load_image(annotation_path)
            annotation = one_hot_to_image(
                image_to_one_hot(annotation, self.class_colors))

            # Todo: check if this is the correct this to do.
            annotation = ia.SegmentationMapOnImage(
                annotation, shape=image.shape, nb_classes=21)
            annotations_batch.append(annotation)

        batch = ia.Batch(
            images=np.array(images_batch, dtype=np.float32),
            segmentation_maps=annotations_batch)

        self.current_step += 1

        self._check_epoch_end()

        yield batch

    def _check_epoch_end(self):
        if self.current_step == self.steps_per_epoch:
            print('End of epoch {}'.format(self.current_epoch))
            self.current_epoch += 1
            self.current_step = 0
            self._shuffle_paths()


augmentation_pipeline = augmenters.Sequential([
    augmenters.OneOf([
        augmenters.Fog(deterministic=True),
        augmenters.Snowflakes(deterministic=True),
        augmenters.FastSnowyLandscape(deterministic=True),
        augmenters.GaussianBlur(sigma=(0.0, 2.0), deterministic=True),
        augmenters.Add((-20, 20), per_channel=0.5, deterministic=True),
        augmenters.CoarseDropout(
            0.02, size_percent=0.05, per_channel=0.5, deterministic=True),
        augmenters.AdditiveGaussianNoise(scale=0.1 * 255, deterministic=True),
        augmenters.AddElementwise(
            (-10, 10), per_channel=0.5, deterministic=True),
        augmenters.Emboss(
            alpha=(0.0, 0.5), strength=(0.5, 1.5), deterministic=True)
    ]),
    augmenters.SomeOf(3, [
        augmenters.Grayscale(alpha=(0.0, 1.0)),
        augmenters.ContrastNormalization((0.5, 1.5)),
        augmenters.Affine(scale=(0.5, 1.5)),
        augmenters.Affine(rotate=(-20, 20)),
        augmenters.CropAndPad(percent=(-0.15, 0.15)),
        augmenters.Sharpen((0.0, 1.0)),
    ])
],
    deterministic=True)


if __name__ == "__main__":
    # Todo: finish this.
    data_generator = DataGenerator(2, 1)
    batch_loader = ia.BatchLoader(data_generator)
    bg_augmenter = ia.BackgroundAugmenter(batch_loader, )

    while True:
        batch = bg_augmenter.get_batch()

        if batch is None:
            print('Went through dataset.')
            break

        augmented_images = batch.images_aug
        augmented_annotations = batch.segmentation_maps_aug

    batch_loader.terminate()
    bg_augmenter.terminate()
