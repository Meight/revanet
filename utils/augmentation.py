""" Various data augmentation utils.
"""
import random
import cv2
import tensorflow as tf


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


def resize_to_fixed_size(img_tensor,
                         annotation_tensor,
                         output_shape,
                         min_relative_random_scale_change=0.9,
                         max_realtive_random_scale_change=1.1,
                         mask_out_number=255):
    """Returns tensor of a size (output_shape, output_shape, depth) and (output_shape, output_shape, 1).
    The function returns tensor that is of a size (output_shape, output_shape, depth)
    which is randomly scaled by a factor that is sampled from a uniform distribution
    between values [min_relative_random_scale_change, max_realtive_random_scale_change] multiplied
    by the factor that is needed to scale image to the output_shape. When the rescaled image
    doesn't fit into the [output_shape] size, the image is either padded or cropped. Also, the
    function returns scaled annotation tensor of the size (output_shape, output_shape, 1). Both,
    the image tensor and the annotation tensor are scaled using nearest neighbour interpolation.
    This was done to preserve the annotation labels. Be careful when specifying the big sample
    space for the random variable -- aliasing effects can appear. When scaling, this function
    preserves the aspect ratio of the original image. When performing all of those manipulations
    there will be some regions in the output image with blank regions -- the function masks out
    those regions in the annotation using mask_out_number. Overall, the function performs the
    rescaling neccessary to get image of output_shape, adds random scale jitter, preserves
    scale ratio, masks out unneccassary regions that appear.

    Parameters
    ----------
    img_tensor : Tensor of size (width, height, depth)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with respective annotation
    output_shape : Tensor or list [int, int]
        Tensor of list representing desired output shape
    min_relative_random_scale_change : float
        Lower bound for uniform distribution to sample from
        when getting random scaling jitter
    max_realtive_random_scale_change : float
        Upper bound for uniform distribution to sample from
        when getting random scaling jitter
    mask_out_number : int
        Number representing the mask out value.

    Returns
    -------
    cropped_padded_img : Tensor of size (output_shape[0], output_shape[1], 3).
        Image Tensor that was randomly scaled
    cropped_padded_annotation : Tensor of size (output_shape[0], output_shape[1], 1)
        Respective annotation Tensor that was randomly scaled with the same parameters
    """

    # tf.image.resize_nearest_neighbor needs
    # first dimension to represent the batch number
    img_batched = tf.expand_dims(img_tensor, 0)
    annotation_batched = tf.expand_dims(annotation_tensor, 0)

    # Convert to int_32 to be able to differentiate
    # between zeros that was used for padding and
    # zeros that represent a particular semantic class
    annotation_batched = tf.to_int32(annotation_batched)

    # Get height and width tensors
    input_shape = tf.shape(img_batched)[1:3]

    input_shape_float = tf.to_float(input_shape)

    scales = output_shape / input_shape_float

    rand_var = tf.random_uniform(shape=[1],
                                 minval=min_relative_random_scale_change,
                                 maxval=max_realtive_random_scale_change)

    final_scale = tf.reduce_min(scales) * rand_var

    scaled_input_shape = tf.to_int32(tf.round(input_shape_float * final_scale))

    # Resize the image and annotation using nearest neighbour
    # Be careful -- may cause aliasing.

    resized_img = tf.image.resize_nearest_neighbor(img_batched, scaled_input_shape)
    resized_annotation = tf.image.resize_nearest_neighbor(annotation_batched, scaled_input_shape)

    resized_img = tf.squeeze(resized_img, axis=0)
    resized_annotation = tf.squeeze(resized_annotation, axis=0)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    annotation_shifted_classes = resized_annotation + 1

    cropped_padded_img = tf.image.resize_image_with_crop_or_pad(resized_img, output_shape[0], output_shape[1])

    cropped_padded_annotation = tf.image.resize_image_with_crop_or_pad(annotation_shifted_classes,
                                                                       output_shape[0],
                                                                       output_shape[1])

    annotation_additional_mask_out = tf.to_int32(tf.equal(cropped_padded_annotation, 0)) * (mask_out_number + 1)

    cropped_padded_annotation = cropped_padded_annotation + annotation_additional_mask_out - 1

    return cropped_padded_img, cropped_padded_annotation
