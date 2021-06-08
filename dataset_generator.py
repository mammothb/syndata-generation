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
from util_bbox import overlap
from util_image import (
    get_annotation_from_mask,
    get_annotation_from_mask_file,
    linear_motion_blur_3c,
    pil_to_array_1c,
    pil_to_array_3c,
)
from util_io import (
    get_list_of_images,
    get_mask_file,
    write_imageset_file,
    write_labels_file,
)

sys.path.insert(0, POISSON_BLENDING_DIR)
import pb

Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")
CWD = Path(__file__).resolve().parent


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


def create_image_anno_wrapper(
    args,
    conf,
    scale_augment=False,
    rotate_augment=False,
    blending_list=["none"],
    no_occlusion=False,
):
    """Wrapper used to pass params to workers"""
    return create_image_anno(
        *args,
        conf,
        scale_augment=scale_augment,
        rotate_augment=rotate_augment,
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
    rotate_augment=False,
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
        rotate_augment(bool): Add rotation data augmentation
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
        for _ in blending_list:
            backgrounds.append(background.copy())

        # Potential forced occlusion
        # 1. Chance
        # 2. Position
        # 3. Size/scale
        # 4. Pre-occluded images
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
                mask = Image.fromarray(255 - pil_to_array_1c(mask)).convert("1")
            o_w, o_h = orig_w, orig_h
            if scale_augment:
                while True:
                    scale = random.uniform(conf["min_scale"], conf["max_scale"])
                    o_w, o_h = int(scale * orig_w), int(scale * orig_h)
                    if w - o_w > 0 and h - o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
            if rotate_augment:
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
                    img_mask = pil_to_array_1c(mask)
                    img_src = pil_to_array_3c(foreground).astype(np.float64)
                    img_target = pil_to_array_3c(backgrounds[i])
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
                        Image.fromarray(
                            cv2.GaussianBlur(pil_to_array_1c(mask), (5, 5), 2)
                        ),
                    )
                elif blending_list[i] == "box":
                    backgrounds[i].paste(
                        foreground,
                        (x, y),
                        Image.fromarray(cv2.blur(pil_to_array_1c(mask), (3, 3))),
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
            backgrounds[i] = linear_motion_blur_3c(pil_to_array_3c(backgrounds[i]))
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
    rotate_augment,
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
        rotate_augment(bool): Add rotation data augmentation
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

    # partial_func = partial(
    #     create_image_anno_wrapper,
    #     conf=conf,
    #     scale_augment=scale_augment,
    #     rotate_augment=rotate_augment,
    #     blending_list=conf["blending"],
    #     no_occlusion=no_occlusion,
    # )
    # p = Pool(conf["num_workers"], init_worker)
    # try:
    #     p.map(partial_func, params_list)
    # except KeyboardInterrupt:
    #     print("....\nCaught KeyboardInterrupt, terminating workers")
    #     p.terminate()
    # else:
    #     p.close()
    # p.join()
    for params in params_list:
        create_image_anno_wrapper(
            params,
            conf=conf,
            scale_augment=scale_augment,
            rotate_augment=rotate_augment,
            blending_list=conf["blending"],
            no_occlusion=no_occlusion,
        )

    return img_files, anno_files


def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def generate_synthetic_dataset(args):
    """Generate synthetic dataset according to given args"""
    img_files = get_list_of_images(CWD / args.root, args.num)
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
        help=(
            "Keep only selected instances in the test dataset. "
            "Default is to keep all instances in the root directory"
        ),
        action="store_true",
    )
    parser.add_argument(
        "--scale",
        help="Add scale augmentation. Default is to add scale augmentation.",
        action="store_false",
    )
    parser.add_argument(
        "--rotation",
        help="Add rotation augmentation. Default is to add rotation augmentation.",
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
