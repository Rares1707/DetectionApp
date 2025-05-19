import os

import pydicom
import cv2

import torch
from torchvision.transforms import v2

from .image_data import ImageData
from utils.project_config import DEVICE


def get_maps(jpg_directory):
    names = [file_name.strip() for file_name in sorted(os.listdir(jpg_directory))]
    name_to_index = {}
    index_to_name = {}
    for i, el in enumerate(sorted(names)):
        name_to_index[el] = i
        index_to_name[i] = el
    return name_to_index, index_to_name


def parse_csv(csv_path, jpg_image_directory, index_to_name):
    images = []
    previous_image_name = "please don't be an image name"  # TODO: fix this
    image_index = -1
    with open(csv_path, "r") as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            line = line.replace('"', "")
            tokens = line.split(",")

            if (
                previous_image_name != tokens[0].strip()
            ):  # this is tricky because the names are from jpegs, not from dicoms
                previous_image_name = tokens[0].strip()
                image_index += 1

            region_type_token = tokens[5]
            if region_type_token == "{}":
                continue

            image_name = index_to_name[image_index]
            image_path = os.path.join(jpg_image_directory, image_name)

            region_type = region_type_token.strip().split(":")[1]
            cx = int(tokens[6].strip().split(":")[1])
            cy = int(tokens[7].strip().split(":")[1])
            r = float(tokens[8].strip("}").split(":")[1])

            if region_type == "circle":
                image_data = ImageData(image_path, None, None, region_type, cx, cy, r)
            elif region_type == "ellipse":
                rx = r
                ry = float(tokens[9].strip().split(":")[1])
                theta = float(tokens[10].strip("}").split(":")[1])
                image_data = ImageData(
                    image_path, None, None, region_type, cx, cy, rx, ry, theta
                )

            images.append(image_data)
    return images


def get_corresponding_dicom_image(jpg_image_path, dicom_directory, jpg_name_to_index):
    index = jpg_name_to_index[os.path.basename(jpg_image_path)]
    dicom_names = [x.strip() for x in sorted(os.listdir(dicom_directory))]
    dicom_name = dicom_names[index]
    dicom_path = os.path.join(dicom_directory, dicom_name)
    return dicom_path


def get_ratio(jpg_directory, dicom_directory):
    jpg_names = [x.strip() for x in sorted(os.listdir(jpg_directory))]
    dicom_names = [x.strip() for x in sorted(os.listdir(dicom_directory))]
    jpg_image = cv2.imread(os.path.join(jpg_directory, jpg_names[0]))
    dicom_image = pydicom.dcmread(
        os.path.join(dicom_directory, dicom_names[0])
    ).pixel_array
    return jpg_image.shape[0] / dicom_image.shape[0]
