import math
import sys

import cv2
import numpy as np
from PIL import Image

from defaults import PYBLUR_DIR

sys.path.insert(0, PYBLUR_DIR)
import pyblur


def get_annotation_from_mask(mask):
    """Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1


def get_annotation_from_mask_file(mask_file, inverted_mask, scale=1.0):
    """Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    """
    if mask_file.exists():
        mask = cv2.imread(str(mask_file))
        if inverted_mask:
            mask = 255 - mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return (
                int(scale * xmin),
                int(scale * xmax),
                int(scale * ymin),
                int(scale * ymax),
            )
        else:
            return -1, -1, -1, -1
    else:
        print(f"{mask_file} not found. Using empty mask instead.")
        return -1, -1, -1, -1


def linear_motion_blur_3c(img):
    """Performs motion blur on an image with 3 channels. Used to simulate
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    line_lengths = [3, 5, 7, 9]
    line_types = ["right", "left", "full"]
    line_length_idx = np.random.randint(0, len(line_lengths))
    line_type_idx = np.random.randint(0, len(line_types))
    line_length = line_lengths[line_length_idx]
    line_type = line_types[line_type_idx]
    line_angle = random_angle(line_length)
    blurred_img = img
    for i in range(3):
        blurred_img[:, :, i] = pil_to_array_1c(
            pyblur.LinearMotionBlur(img[:, :, i], line_length, line_angle, line_type)
        )
    blurred_img = Image.fromarray(blurred_img, "RGB")
    return blurred_img


def pil_to_array_1c(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])


def pil_to_array_3c(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def random_angle(kernel_dim):
    """Returns a random angle used to produce motion blurring

    Args:
        kernel_dim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """
    kernel_center = int(math.floor(kernel_dim / 2))
    num_lines = kernel_center * 4
    valid_line_angles = np.linspace(0, 180, num_lines, endpoint=False)
    angle_idx = np.random.randint(0, len(valid_line_angles))
    return int(valid_line_angles[angle_idx])
