from utils.utils import resize_to_size


def data_augmentation(input_image, output_image, input_size):
    input_image, output_image = resize_to_size(input_image, output_image, input_size)

    # TODO: implement actual data augmentation.

    return input_image, output_image
