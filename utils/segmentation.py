import numpy as np


def image_to_one_hot(annotation, class_colors):
    semantic_map = []
    for color in class_colors:
        equality = np.equal(annotation, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def apply_threshold_to_prediction(prediction, threshold=0.5):
    assert 0 <= threshold <= 1

    accepted_predictions_indices = np.where(prediction >= threshold)
    x = np.zeros_like(prediction)
    x[accepted_predictions_indices] = 1.0

    return x


def one_hot_to_image(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, class_colors):
    return np.array(class_colors)[image.astype(int)]
