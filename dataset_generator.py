import argparse
import random
import signal
from collections import namedtuple
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import yaml
from PIL import Image

from defaults import CONFIG_FILE
from util_bbox import overlap
from util_image import (
    add_localized_distractor,
    blend_object,
    get_annotation_from_mask,
    get_annotation_from_mask_file,
    invert_mask,
    linear_motion_blur_3c,
    perspective_transform,
    pil_to_array_3c,
    scale_object,
    rotate_object,
)
from util_io import (
    get_labels,
    get_occlusion_coords,
    get_list_of_images,
    get_mask_file,
    print_paths,
    write_imageset_file,
    write_labels_file,
)


Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")
CWD = Path(__file__).resolve().parent
# SEED = 123
# random.seed(SEED)


def keep_selected_labels(img_files, labels, occ_coords, conf):
    """Filters image files and labels to only retain those that are selected. Useful when one doesn't
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
        conf(dict): Config options
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponding to each image in above list
    """
    new_img_files = []
    new_labels = []
    new_occ_coords = []
    for i, img_file in enumerate(img_files):
        if labels[i] in conf["selected"]:
            new_img_files.append(img_file)
            new_labels.append(labels[i])
            new_occ_coords.append(occ_coords[i])
    return new_img_files, new_labels, new_occ_coords


def create_image_anno_wrapper(
    args, conf, opt, blending_list=["none"],
):
    """Wrapper used to pass params to workers"""
    return create_image_anno(*args, conf, opt, blending_list=blending_list)


def constrained_randint(frac, fg_dim, bg_dim):
    """Return a random int constrained by the allowed trunction fraction and
    bg/fg sizes

    Args:
        frac(float): Max allowed truncation fraction
        fg_dim(int): Foreground dimension (height or width)
        bg_dim(int): Background dimension (height or width)
    """
    return random.randint(int(-frac * fg_dim), int(bg_dim - fg_dim + frac * fg_dim))


def constrained_rand_num(obj_type, max_num, conf):
    return min(
        random.randint(conf[f"min_{obj_type}_num"], conf[f"max_{obj_type}_num"]),
        max_num,
    )


def create_image_anno(
    objects,
    distractor_objects,
    localized_distractors,
    img_file,
    anno_file,
    bg_file,
    conf,
    opt,
    blending_list=["none"],
):
    """Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path
        conf(dict): Config options
        opt(Namespace): Contains options to:
                1. Add scale data augmentation
                2. Add rotation data augmentation
                3. Generate images with occlusion
                4. Add distractor objects whose annotations are not required
        blending_list(list): List of blending modes to synthesize for each image
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

        if not opt.occlude:
            already_syn = []
        for idx, obj in enumerate(all_objects):
            foreground = Image.open(obj[0])
            mask_file = get_mask_file(obj[0])
            xmin, xmax, ymin, ymax = get_annotation_from_mask_file(
                mask_file, conf["inverted_mask"]
            )
            mask = Image.open(mask_file)
            if conf["inverted_mask"]:
                mask = invert_mask(mask)
            mask_bb = mask.copy()
            if (
                xmin == -1
                or ymin == -1
                or xmax - xmin < conf["min_width"]
                or ymax - ymin < conf["min_height"]
            ):
                continue
            # foreground_crop = foreground.crop((xmin, ymin, xmax, ymax))
            # orig_w, orig_h = foreground_crop.size
            orig_w, orig_h = xmax - xmin, ymax - ymin
            if idx < len(objects) and localized_distractors[idx]:
                for distractor in localized_distractors[idx]:
                    foreground, mask, mask_bb = add_localized_distractor(
                        distractor,
                        foreground.size,
                        (orig_w, orig_h),
                        conf,
                        opt,
                        foreground,
                        mask,
                        mask_bb,
                    )
                xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            foreground = foreground.crop((xmin, ymin, xmax, ymax))
            mask = mask.crop((xmin, ymin, xmax, ymax))
            mask_bb = mask_bb.crop((xmin, ymin, xmax, ymax))
            orig_w, orig_h = foreground.size
            rel_scale = 1
            if orig_w > orig_h and orig_w > w * 0.75:
                rel_scale = w * 0.75 / orig_w
            elif orig_h > orig_w and orig_h > h * 0.75:
                rel_scale = h * 0.75 / orig_h
            orig_w, orig_h = int(orig_w * rel_scale), int(orig_h * rel_scale)
            foreground = foreground.resize((orig_w, orig_h), Image.ANTIALIAS)
            mask = mask.resize((orig_w, orig_h), Image.ANTIALIAS)
            mask_bb = mask_bb.resize((orig_w, orig_h), Image.ANTIALIAS)
            o_w, o_h = orig_w, orig_h
            if opt.scale:
                foreground, mask, mask_bb = scale_object(
                    foreground, mask, mask_bb, h, w, orig_h, orig_w, conf
                )
            if opt.rotate:
                foreground, mask, mask_bb = rotate_object(
                    foreground, mask, mask_bb, h, w, conf
                )
                o_w, o_h = foreground.size
            if opt.perspective:
                foreground, mask, mask_bb = perspective_transform(
                    foreground, mask, mask_bb, o_h, o_w, conf
                )
                o_w, o_h = foreground.size
            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask_bb)
            attempt = 0
            while True:
                attempt += 1
                x = constrained_randint(conf["max_truncation_frac"], o_w, w)
                y = constrained_randint(conf["max_truncation_frac"], o_h, h)
                if opt.occlude:
                    break
                found = True
                for prev in already_syn:
                    ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                    rb = Rectangle(x + xmin, y + ymin, x + xmax, y + ymax)
                    if overlap(ra, rb, conf["max_allowed_iou"]):
                        found = False
                        break
                if found:
                    break
                if attempt == conf["max_attempts"]:
                    break
            if not opt.occlude:
                already_syn.append([x + xmin, x + xmax, y + ymin, y + ymax])
            for i, blending in enumerate(blending_list):
                backgrounds[i] = blend_object(
                    blending, backgrounds[i], foreground, mask, x, y
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
    for i, blending in enumerate(blending_list):
        if blending == "motion":
            backgrounds[i] = linear_motion_blur_3c(pil_to_array_3c(backgrounds[i]))
        backgrounds[i].save(str(img_file).replace("none", blending))

    with open(anno_file, "w") as f:
        f.write("\n".join(boxes))


def gen_syn_data(
    img_files, labels, occ_coords, img_dir, anno_dir, conf, opt,
):
    """Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
        conf(dict): Config options
        opt(Namespace): Contains options to:
                        1. Add scale data augmentation
                        2. Add rotation data augmentation
                        3. Generate images with occlusion
                        4. Add distractor objects whose annotations are not required
    """
    background_files = list(
        (CWD / conf["background_dir"]).glob(conf["background_glob"])
    )

    print(f"Number of background images: {len(background_files)}")
    img_labels = list(zip(img_files, labels, occ_coords))
    random.shuffle(img_labels)

    if opt.distract:
        distractor_list = []
        for label in conf["distractor"]:
            distractor_list += list(
                (CWD / conf["distractor_dir"] / label).glob(conf["distractor_glob"])
            )

        distractor_files = list(zip(distractor_list, len(distractor_list) * [None]))
        random.shuffle(distractor_files)
    else:
        distractor_files = []
    print_paths("List of distractor files collected:", distractor_files)

    idx = 0
    img_files = []
    anno_files = []
    params_list = []
    while len(img_labels) > 0:
        # Get list of objects
        objects = []
        n = constrained_rand_num("object", len(img_labels), conf)
        for _ in range(n):
            objects.append(img_labels.pop())
        # Get list of distractor objects
        distractor_objects = []
        if opt.distract:
            n = constrained_rand_num("distractor", len(distractor_files), conf)
            for _ in range(n):
                distractor_objects.append(random.choice(distractor_files))
            print_paths("Chosen distractor objects:", distractor_objects)

        localized_distractors = []
        if opt.localized_distractor:
            for obj in objects:
                localized_distractors.append(
                    [
                        (random.choice(distractor_files), coord)
                        for coord in obj[2]
                        if random.random() < 1.0
                    ]
                )

        idx += 1
        bg_file = random.choice(background_files)
        for blur in conf["blending"]:
            img_file = img_dir / f"{idx}_{blur}.jpg"
            anno_file = anno_dir / f"{idx}.txt"
            params = (
                objects,
                distractor_objects,
                localized_distractors,
                img_file,
                anno_file,
                bg_file,
            )
            params_list.append(params)
            img_files.append(img_file)
            anno_files.append(anno_file)

    partial_func = partial(
        create_image_anno_wrapper, conf=conf, opt=opt, blending_list=conf["blending"],
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
    # for params in params_list:
    #     create_image_anno_wrapper(
    #         params, conf=conf, opt=opt, blending_list=conf["blending"],
    #     )

    return img_files, anno_files


def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def generate_synthetic_dataset(opt):
    """Generate synthetic dataset according to given args"""
    img_files = get_list_of_images(CWD / opt.root, opt.num)
    labels = get_labels(img_files)
    occ_coords = get_occlusion_coords(img_files)

    with open(CONFIG_FILE, "r") as infile:
        conf = yaml.safe_load(infile)
    if opt.selected:
        img_files, labels, occ_coords = keep_selected_labels(
            img_files, labels, occ_coords, conf
        )

    exp_dir = CWD / opt.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    write_labels_file(exp_dir, labels)

    anno_dir = exp_dir / "annotations"
    img_dir = exp_dir / "images"
    anno_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    syn_img_files, anno_files = gen_syn_data(
        img_files, labels, occ_coords, img_dir, anno_dir, conf, opt
    )
    write_imageset_file(exp_dir, syn_img_files, anno_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset with augmentations")
    parser.add_argument("root", help="Root directory containing images and annotations")
    parser.add_argument("exp", help="Output directory for images and annotations")
    parser.add_argument(
        "--selected",
        help="Keep only selected instances in the test dataset. Default False",
        action="store_true",
    )
    parser.add_argument(
        "--distract", help="Add distractors objects. Default False", action="store_true"
    )
    parser.add_argument(
        "--occlude",
        help="Allow objects with full occlusion. Default False",
        action="store_true",
    )
    parser.add_argument(
        "--perspective",
        help="Add perspective transform. Default True",
        action="store_false",
    )
    parser.add_argument(
        "--rotate", help="Add rotation augmentation. Default True", action="store_false"
    )
    parser.add_argument(
        "--scale", help="Add scale augmentation. Default True", action="store_false"
    )
    parser.add_argument(
        "--num",
        help="Number of times each image will be in dataset",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--localized_distractor",
        help="Add occluding distractors to specified spots. Default False",
        action="store_true",
    )
    opt = parser.parse_args()
    generate_synthetic_dataset(opt)
