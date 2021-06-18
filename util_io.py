import random

import yaml


def get_labels(imgs):
    """Get list of labels/object names. Assumes the images in the root directory
    follow root_dir/<object>/<image> structure. Directory name would be object
    name.

    Args:
        imgs(list): List of images being used for synthesis
    Returns:
        list: List of labels/object names corresponding to each image
    """
    return [img_file.parent.stem for img_file in imgs]


def get_occlusion_coords(imgs):
    """Get list of coordinates for localized distractor in each image. 
    """
    occlusion_coords = []
    curr_label = None
    for img_file in imgs:
        if curr_label != img_file.parent:
            curr_label = img_file.parent
            try:
                with open(img_file.parent / "occlusion_coords.yaml") as infile:
                    coords_data = yaml.safe_load(infile)
            except FileNotFoundError:
                coords_data = {}
        occlusion_coords.append(coords_data.get(img_file.name, []))
    return occlusion_coords


def get_list_of_images(root_dir, num):
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
    img_list = list(root_dir.glob("*/*.jpg"))
    img_list_f = []
    for _ in range(num):
        img_list_f += random.sample(img_list, len(img_list))
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


def print_paths(header, paths):
    print(header)
    for path, _ in paths:
        print(f"  {path}")


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
