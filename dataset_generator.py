import argparse
import glob
import math
import os
import random
import signal
import sys
from collections import namedtuple
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import scipy
import yaml
from PIL import Image

from defaults import CONFIG_FILE, POISSON_BLENDING_DIR, PYBLUR_DIR

sys.path.insert(0, POISSON_BLENDING_DIR)
sys.path.insert(0, PYBLUR_DIR)
import pb
import pyblur

Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

CWD = Path(__file__).resolve().parent


def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """
    kernelCenter = int(math.floor(kerneldim / 2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])


def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3, 5, 7, 9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:, :, i] = PIL2array1C(
            pyblur.LinearMotionBlur(img[:, :, i], lineLength, lineAngle, lineType)
        )
    blurred_img = Image.fromarray(blurred_img, "RGB")
    return blurred_img


def overlap(a, b, max_allowed_iou):
    """Find if two bounding boxes are overlapping or not. This is determined by maximum allowed
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    """
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    if (
        dx >= 0
        and dy >= 0
        and float(dx * dy) > max_allowed_iou * (a.xmax - a.xmin) * (a.ymax - a.ymin)
    ):
        return True
    else:
        return False


def get_list_of_images(root_dir, N=1):
    """Gets the list of images of objects in the root directory. The expected format
       is root_dir/<object>/<image>.jpg. Adds an image as many times you want it to
       appear in dataset.

    Args:
        root_dir(string): Directory where images of objects are present
        N(int): Number of times an image would appear in dataset. Each image should have
                different data augmentation
    Returns:
        list: List of images(with paths) that will be put in the dataset
    """
    img_list = list((Path(__file__).resolve().parent / root_dir).glob("*/*.jpg"))
    img_list_f = []
    for _ in range(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f


def get_mask_file(img_file):
    """Takes an image file name and returns the corresponding mask file. The mask represents
       pixels that belong to the object. Default implentation assumes mask file has same path
       as image file with different extension only. Write custom code for getting mask file here
       if this is not the case.

    Args:
        img_file(string): Image name
    Returns:
        string: Correpsonding mask file path
    """
    return img_file.with_suffix(".pbm")


def get_labels(imgs):
    """Get list of labels/object names. Assumes the images in the root directory follow root_dir/<object>/<image>
       structure. Directory name would be object name.

    Args:
        imgs(list): List of images being used for synthesis
    Returns:
        list: List of labels/object names corresponding to each image
    """
    labels = [img_file.parent.stem for img_file in imgs]
    return labels


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


def write_imageset_file(exp_dir, img_files, anno_files):
    """Writes the imageset file which has the generated images and corresponding annotation files
       for a given experiment

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        img_files(list): List of image files that were generated
        anno_files(list): List of annotation files corresponding to each image file
    """
    with open(exp_dir / "train.txt", "w") as f:
        for i in range(len(img_files)):
            f.write(f"{img_files[i]} {anno_files[i]}\n")


def write_labels_file(exp_dir, labels):
    """Writes the labels file which has the name of an object on each line

    Args:
        exp_dir(Path): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    """
    unique_labels = ["__background__"] + sorted(set(labels))
    with open(exp_dir / "labels.txt", "w") as f:
        for i, label in enumerate(unique_labels):
            f.write("%s %s\n" % (i, label))


def keep_selected_labels(img_files, labels, conf):
    """Filters image files and labels to only retain those that are selected. Useful when one doesn't
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
        conf(dict) : Config options
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponding to each image in above list
    """
    new_img_files = []
    new_labels = []
    for i in range(len(img_files)):
        if labels[i] in conf["selected"]:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels


def PIL2array1C(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])


def PIL2array3C(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def create_image_anno_wrapper(
    args,
    conf,
    scale_augment=False,
    rotation_augment=False,
    blending_list=["none"],
    no_occlusion=False,
):
    """Wrapper used to pass params to workers"""
    return create_image_anno(
        *args,
        conf,
        scale_augment=scale_augment,
        rotation_augment=rotation_augment,
        blending_list=blending_list,
        no_occlusion=no_occlusion,
    )


def create_image_anno(
    objects,
    distractor_objects,
    img_file,
    anno_file,
    bg_file,
    conf,
    scale_augment=False,
    rotation_augment=False,
    blending_list=["none"],
    no_occlusion=False,
):
    """Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        no_occlusion(bool): Generate images with occlusion
    """
    if "none" not in img_file.name:
        return

    print(f"Working on {img_file}")
    if anno_file.exists():
        return anno_file

    all_objects = objects + distractor_objects
    assert len(all_objects) > 0

    w = conf["width"]
    h = conf["height"]
    while True:
        boxes = []
        background = Image.open(bg_file)
        background = background.resize((w, h), Image.ANTIALIAS)
        backgrounds = []
        for i in range(len(blending_list)):
            backgrounds.append(background.copy())

        if no_occlusion:
            already_syn = []
        for idx, obj in enumerate(all_objects):
            foreground = Image.open(obj[0])
            mask_file = get_mask_file(obj[0])
            xmin, xmax, ymin, ymax = get_annotation_from_mask_file(
                mask_file, conf["inverted_mask"]
            )
            if (
                xmin == -1
                or ymin == -1
                or xmax - xmin < conf["min_width"]
                or ymax - ymin < conf["min_height"]
            ):
                continue
            foreground = foreground.crop((xmin, ymin, xmax, ymax))
            orig_w, orig_h = foreground.size
            mask = Image.open(mask_file)
            mask = mask.crop((xmin, ymin, xmax, ymax))
            if conf["inverted_mask"]:
                mask = Image.fromarray(255 - PIL2array1C(mask)).convert("1")
            o_w, o_h = orig_w, orig_h
            if scale_augment:
                while True:
                    scale = random.uniform(conf["min_scale"], conf["max_scale"])
                    o_w, o_h = int(scale * orig_w), int(scale * orig_h)
                    if w - o_w > 0 and h - o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
            if rotation_augment:
                while True:
                    rot_degrees = random.randint(
                        -conf["max_degrees"], conf["max_degrees"]
                    )
                    foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                    mask_tmp = mask.rotate(rot_degrees, expand=True)
                    o_w, o_h = foreground_tmp.size
                    if w - o_w > 0 and h - o_h > 0:
                        break
                mask = mask_tmp
                foreground = foreground_tmp
            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            attempt = 0
            while True:
                attempt += 1
                x = random.randint(
                    int(-conf["max_truncation_frac"] * o_w),
                    int(w - o_w + conf["max_truncation_frac"] * o_w),
                )
                y = random.randint(
                    int(-conf["max_truncation_frac"] * o_h),
                    int(h - o_h + conf["max_truncation_frac"] * o_h),
                )
                if no_occlusion:
                    found = True
                    for prev in already_syn:
                        ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                        rb = Rectangle(x + xmin, y + ymin, x + xmax, y + ymax)
                        if overlap(ra, rb, conf["max_allowed_iou"]):
                            found = False
                            break
                    if found:
                        break
                else:
                    break
                if attempt == conf["max_attempts"]:
                    break
            if no_occlusion:
                already_syn.append([x + xmin, x + xmax, y + ymin, y + ymax])
            for i in range(len(blending_list)):
                if blending_list[i] == "none" or blending_list[i] == "motion":
                    backgrounds[i].paste(foreground, (x, y), mask)
                elif blending_list[i] == "poisson":
                    offset = (y, x)
                    img_mask = PIL2array1C(mask)
                    img_src = PIL2array3C(foreground).astype(np.float64)
                    img_target = PIL2array3C(backgrounds[i])
                    img_mask, img_src, offset_adj = pb.create_mask(
                        img_mask.astype(np.float64), img_target, img_src, offset=offset
                    )
                    background_array = pb.poisson_blend(
                        img_mask,
                        img_src,
                        img_target,
                        method="normal",
                        offset_adj=offset_adj,
                    )
                    backgrounds[i] = Image.fromarray(background_array, "RGB")
                elif blending_list[i] == "gaussian":
                    backgrounds[i].paste(
                        foreground,
                        (x, y),
                        Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask), (5, 5), 2)),
                    )
                elif blending_list[i] == "box":
                    backgrounds[i].paste(
                        foreground,
                        (x, y),
                        Image.fromarray(cv2.blur(PIL2array1C(mask), (3, 3))),
                    )
            if idx >= len(objects):
                continue
            x_min = max(1, x + xmin)
            x_max = min(w, x + xmax)
            y_min = max(1, y + ymin)
            y_max = min(h, y + ymax)
            boxes.append(
                f"{str(obj[1])} "
                f"{(x_min + x_max) / 2 / w} "
                f"{(y_min + y_max) / 2 / h} "
                f"{(x_max - x_min) / w} "
                f"{(y_max - y_min) / h}"
            )
        if attempt == conf["max_attempts"]:
            continue
        else:
            break
    for i in range(len(blending_list)):
        if blending_list[i] == "motion":
            backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
        backgrounds[i].save(str(img_file).replace("none", blending_list[i]))

    with open(anno_file, "w") as f:
        f.write("\n".join(boxes))


def gen_syn_data(
    img_files,
    labels,
    img_dir,
    anno_dir,
    conf,
    scale_augment,
    rotation_augment,
    no_occlusion,
    add_distractors,
):
    """Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        no_occlusion(bool): Generate images with occlusion
        add_distractors(bool): Add distractor objects whose annotations are not required
    """
    background_files = list(
        (CWD / conf["background_dir"]).glob(conf["background_glob_string"])
    )

    print(f"Number of background images : {len(background_files)}")
    img_labels = list(zip(img_files, labels))
    random.shuffle(img_labels)

    if add_distractors:
        distractor_list = []
        for distractor_label in conf["distractor"]:
            distractor_list += list(
                (CWD / conf["distractor_dir"] / distractor_label).glob(
                    conf["distractor_glob_string"]
                )
            )

        distractor_files = list(zip(distractor_list, len(distractor_list) * [None]))
        random.shuffle(distractor_files)
    else:
        distractor_files = []
    print(f"List of distractor files collected: {distractor_files}")

    idx = 0
    img_files = []
    anno_files = []
    params_list = []
    while len(img_labels) > 0:
        # Get list of objects
        objects = []
        n = min(
            random.randint(conf["min_object_num"], conf["max_object_num"]),
            len(img_labels),
        )
        for _ in range(n):
            objects.append(img_labels.pop())
        # Get list of distractor objects
        distractor_objects = []
        if add_distractors:
            n = min(
                random.randint(conf["min_distractor_num"], conf["max_distractor_num"]),
                len(distractor_files),
            )
            for _ in range(n):
                distractor_objects.append(random.choice(distractor_files))
            print(f"Chosen distractor objects: {distractor_objects}")

        idx += 1
        bg_file = random.choice(background_files)
        for blur in conf["blending"]:
            img_file = img_dir / f"{idx}_{blur}.jpg"
            anno_file = anno_dir / f"{idx}.txt"
            params = (objects, distractor_objects, img_file, anno_file, bg_file)
            params_list.append(params)
            img_files.append(img_file)
            anno_files.append(anno_file)

    partial_func = partial(
        create_image_anno_wrapper,
        conf=conf,
        scale_augment=scale_augment,
        rotation_augment=rotation_augment,
        blending_list=conf["blending"],
        no_occlusion=no_occlusion,
    )
    p = Pool(conf["num_workers"], init_worker)
    try:
        p.map(partial_func, params_list)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()
    return img_files, anno_files


def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def generate_synthetic_dataset(args):
    """Generate synthetic dataset according to given args"""
    img_files = get_list_of_images(args.root, args.num)
    labels = get_labels(img_files)

    with open(CONFIG_FILE) as f:
        conf = yaml.safe_load(f)
    if args.selected:
        img_files, labels = keep_selected_labels(img_files, labels, conf)

    exp_dir = CWD / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    write_labels_file(exp_dir, labels)

    anno_dir = exp_dir / "annotations"
    img_dir = exp_dir / "images"
    anno_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    syn_img_files, anno_files = gen_syn_data(
        img_files,
        labels,
        img_dir,
        anno_dir,
        conf,
        args.scale,
        args.rotation,
        args.no_occlusion,
        args.add_distractors,
    )
    write_imageset_file(exp_dir, syn_img_files, anno_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dataset with different augmentations"
    )
    parser.add_argument(
        "root", help="The root directory which contains the images and annotations."
    )
    parser.add_argument(
        "exp", help="The directory where images and annotation lists will be created."
    )
    parser.add_argument(
        "--selected",
        help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory",
        action="store_true",
    )
    parser.add_argument(
        "--scale",
        help="Add scale augmentation.Default is to add scale augmentation.",
        action="store_false",
    )
    parser.add_argument(
        "--rotation",
        help="Add rotation augmentation.Default is to add rotation augmentation.",
        action="store_false",
    )
    parser.add_argument(
        "--num",
        help="Number of times each image will be in dataset",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--no_occlusion",
        help="Add objects without occlusion. Default is to produce occlusions",
        action="store_true",
    )
    parser.add_argument(
        "--add_distractors",
        help="Add distractors objects. Default is to not use distractors",
        action="store_true",
    )
    args = parser.parse_args()
    generate_synthetic_dataset(args)
