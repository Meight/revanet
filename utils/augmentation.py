""" Various data augmentation utils.
"""
import random
import cv2


def augment_data(input_image, output_image, input_size):
    """ Applies all the requested transformations to input and output images.

    Encapsulates all the transformations applied to both images and their associated annotation images.

    :param input_image:     The image should be fed to the network.
    :param output_image:    The ground truth image that will be provided to the network.
    :param input_size:      The common size of all the images that will be fed to the network. Both images will
                            be resized if their shape does not comply.
    :return:

    :raises ValueError: if the provided input and the output images don't have the same shape.
    """
    input_image, output_image = resize_to_size(input_image, output_image, input_size)

    # TODO: implement actual data augmentation.

    return input_image, output_image


def resize_to_size(image, label=None, desired_size=256):
    """
    Resizes one or two images to a desired **square** size.

    :param image:           The image to resize.
    :param label:           The (optional) associated annotation image. If provided, this image must have the same shape
                            as the image to resize.
    :param desired_size:    The desired size the images should be resized to.
    :return:                The resized images, both with shapes ``(desired_size, desired_size)``.
    :raises ValueError:     If the annotation image is provided but the shapes mismatch.
    """
    if label is not None and ((image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1])):
        raise Exception('Image and label must have the same dimensions! {} vs {}'.format(image.shape, label.shape))

    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)

    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))

    if label is not None:
        label = cv2.resize(label, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [255, 255, 255]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=color)

    if label is not None:
        label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color)

    return image, label


def crop_randomly(image, annotation, crop_height, crop_width):
    if (image.shape[0] != annotation.shape[0]) or (image.shape[1] != annotation.shape[1]):
        raise ValueError('Image and label must have the same dimensions! {} vs {}'.format(image.shape,
                                                                                          annotation.shape))

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(annotation.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], annotation[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], annotation[y:y + crop_height, x:x + crop_width]
    else:
        raise ValueError('Crop shape ({}, {}) exceeds image dimensions ({}, {}).'.format(crop_height,
                                                                                         crop_width,
                                                                                         image.shape[0],
                                                                                         image.shape[1]))
