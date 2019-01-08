"""
Set of utilities to perform various image processing tasks.
"""

import cv2
import numpy as np


def erode_image(binary_mask, size=5, iterations=1):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(binary_mask, kernel, iterations)


def dilate_image(binary_mask, size=5, iterations=1):
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(binary_mask, kernel, iterations)
