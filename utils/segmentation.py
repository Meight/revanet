import numpy as np


def image_to_one_hot(annotation, class_colors):
    semantic_map = []
    for color in class_colors:
        equality = np.equal(annotation, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def one_hot_to_image(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, class_colors):
    return np.array(class_colors)[image.astype(int)]
