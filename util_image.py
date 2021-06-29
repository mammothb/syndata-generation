import math
import random

import cv2
import numpy as np
from PIL import Image, ImageOps

from pb import pb
from pyblur import pyblur
from util_io import get_mask_file


def add_localized_distractor(
    distractor,
    fg_size,
    fg_crop_size,
    conf,
    opt,
    object_foreground,
    object_mask,
    object_mask_bb,
):
    """
    Args:
        distractor(list): List of distractor objects with their respective
            coordinates
        fg_size(tuple): Object foreground size (width, height)
        fg_crop_size(tuple): Cropped object foreground size (width, height)
        conf(dict): Config options
        opt(Namespace): Contains options to:
                1. Add scale data augmentation
                2. Add rotation data augmentation
                3. Generate images with occlusion
                4. Add distractor objects whose annotations are not required
        object_foreground(PIL.Image): Object foreground
        object_mask(PIL.Image): Object mask
    """
    path = distractor[0][0]
    foreground = Image.open(path)
    mask_file = get_mask_file(path)
    xmin, xmax, ymin, ymax = get_annotation_from_mask_file(
        mask_file, conf["inverted_mask"]
    )
    foreground = foreground.crop((xmin, ymin, xmax, ymax))
    orig_w, orig_h = foreground.size
    mask = Image.open(mask_file)
    mask = mask.crop((xmin, ymin, xmax, ymax))
    if conf["inverted_mask"]:
        mask = invert_mask(mask)
    if orig_w > orig_h:
        o_w = int(fg_crop_size[0] * 0.5)
        o_h = int(orig_h * o_w / orig_w)
    else:
        o_h = int(fg_crop_size[1] * 0.5)
        o_w = int(orig_w * o_h / orig_h)
    foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
    mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
    if opt.rotate:
        rot_degrees = random.randint(-conf["max_degrees"], conf["max_degrees"])
        foreground = foreground.rotate(rot_degrees, expand=True)
        mask = mask.rotate(rot_degrees, expand=True)
        o_w, o_h = foreground.size
    x = int(fg_size[0] * distractor[1][1] - o_w / 2 + 1)
    y = int(fg_size[1] * distractor[1][0] - o_h / 2 + 1)
    pad = -min(x, fg_size[0] - x - o_w, y, fg_size[1] - y - o_h)
    if pad > 0:
        dst_size = (fg_size[0] + 2 * pad, fg_size[1] + 2 * pad)
        object_foreground = ImageOps.pad(object_foreground, dst_size)
        object_mask = ImageOps.pad(object_mask, dst_size)
        object_mask_bb = ImageOps.pad(object_mask_bb, dst_size)
        x += pad
        y += pad
    object_foreground.paste(foreground, (x, y), mask)
    object_mask.paste(mask, (x, y), mask)
    return object_foreground, object_mask, object_mask_bb


def blend_object(blending, background, foreground, mask, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_mask = cv2.erode(pil_to_array_1c(mask), kernel, iterations=1)
    top_left = (x, y)
    if blending == "none" or blending == "motion":
        background.paste(foreground, top_left, Image.fromarray(img_mask))
    elif blending == "poisson":
        offset = (y, x)
        img_src = pil_to_array_3c(foreground).astype(np.float64)
        img_target = pil_to_array_3c(background)
        img_mask, img_src, offset_adj = pb.create_mask(
            img_mask.astype(np.float64), img_target, img_src, offset
        )
        background_array = pb.poisson_blend(
            img_mask, img_src, img_target, method="normal", offset_adj=offset_adj,
        )
        background = Image.fromarray(background_array, "RGB")
    elif blending == "gaussian":
        background.paste(
            foreground,
            top_left,
            Image.fromarray(cv2.GaussianBlur(img_mask, (5, 5), 2)),
        )
    elif blending == "box":
        background.paste(
            foreground, top_left, Image.fromarray(cv2.blur(img_mask, (3, 3))),
        )
    return background


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


def invert_mask(mask):
    return Image.fromarray(255 - pil_to_array_1c(mask)).convert("1")


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


def perspective_transform(foreground, mask, mask_bb, orig_h, orig_w, conf):
    M = np.eye(3)
    # x perspective (about y)
    M[2, 0] = random.uniform(-conf["max_perspective"], conf["max_perspective"])
    # y perspective (about x)
    M[2, 1] = random.uniform(-conf["max_perspective"], conf["max_perspective"])
    coords = np.array([[0, 0], [orig_w, 0], [0, orig_h], [orig_w, orig_h]])
    max_w = 0
    max_h = 0
    for coord in coords:
        denom = M[2, 0] * coord[0] + M[2, 1] * coord[1] + M[2, 2]
        max_w = max(max_w, coord[0] / denom)
        max_h = max(max_h, coord[1] / denom)
    max_w = int(max_w + 1)
    max_h = int(max_h + 1)
    foreground = Image.fromarray(
        cv2.warpPerspective(pil_to_array_3c(foreground), M, dsize=(max_w, max_h))
    )
    mask = Image.fromarray(
        cv2.warpPerspective(pil_to_array_1c(mask), M, dsize=(max_w, max_h))
    )
    mask_bb = Image.fromarray(
        cv2.warpPerspective(
            pil_to_array_1c(mask_bb), M, dsize=(max_w, max_h)
        )
    )
    if max_h > orig_h or max_w > orig_w:
        foreground = foreground.resize((orig_w, orig_h), Image.ANTIALIAS)
        mask = mask.resize((orig_w, orig_h), Image.ANTIALIAS)
        mask_bb = mask_bb.resize((orig_w, orig_h), Image.ANTIALIAS)
    return foreground, mask, mask_bb


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


def rotate_object(foreground, mask, mask_bb, h, w, conf):
    while True:
        rot_degrees = random.randint(-conf["max_degrees"], conf["max_degrees"])
        # foreground_tmp = foreground.rotate(rot_degrees, expand=True)
        mask_tmp = mask.rotate(rot_degrees, expand=True)
        o_w, o_h = mask_tmp.size
        if w - o_w > 0 and h - o_h > 0:
            break
    foreground = foreground.rotate(rot_degrees, expand=True)
    mask_bb = mask_bb.rotate(rot_degrees, expand=True)
    return foreground, mask_tmp, mask_bb


def scale_object(foreground, mask, mask_bb, h, w, orig_h, orig_w, conf):
    while True:
        scale = random.uniform(conf["min_scale"], conf["max_scale"])
        o_w, o_h = int(scale * orig_w), int(scale * orig_h)
        if w - o_w > 0 and h - o_h > 0 and o_w > 0 and o_h > 0:
            break
    return (
        foreground.resize((o_w, o_h), Image.ANTIALIAS),
        mask.resize((o_w, o_h), Image.ANTIALIAS),
        mask_bb.resize((o_w, o_h), Image.ANTIALIAS),
    )

