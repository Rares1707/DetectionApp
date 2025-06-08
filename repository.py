import os

import cv2
import pydicom

from utils.project_config import JPG_FOLDER, DICOM_FOLDER, CSV_HEADER
from utils.classification.data_preparation import (
    get_maps,
    parse_csv,
    get_corresponding_dicom_image,
    get_ratio,
)


class Repository:
    def __init__(self):
        self._output_folder = None
        self._input_folder = None
        self._csv_file = None

        # used only in detection
        self._image_files = []

        # used only for classification
        self.dicom_directory = None
        self.jpg_directory = None
        self.csv_file = None
        self.jpg_name_to_index = None
        self.index_to_jpg_name = None
        self.all_image_data = []

    def output_folder_is_set(self):
        return self._output_folder is not None

    def get_output_image_name(self, image_index):
        return f"{image_index + 1}.png"

    def prepare_input_files_for_detection(self):
        self.all_image_data = []

        self._image_files = [
            file_name
            for file_name in os.listdir(self._input_folder)
            if file_name.lower().endswith((".dcm"))
        ]

        if len(self._image_files) == 0:
            raise ValueError("No DICOM files found in the input folder.")
        self._image_files.sort()

    def csv_is_valid(self):
        if self.csv_file is None or not os.path.isfile(self.csv_file):
            return False

        with open(self.csv_file, "r") as f:
            header = f.readline().strip()
            if header != CSV_HEADER:
                return False
        return True

    def prepare_input_files_for_classification(self):
        csv_files_found = [
            file
            for file in os.listdir(self._input_folder)
            if file.lower().endswith(".csv")
        ]
        if len(csv_files_found) != 1:
            raise ValueError(
                f"There should be exactly ONE csv file in the input folder. You provided: {len(csv_files_found)}"
            )
        self.csv_file = os.path.join(self._input_folder, csv_files_found[0])
        if not self.csv_is_valid():
            raise ValueError(
                f"Error while parsing the CSV file, make sure the CSV file has the correct format."
            )

        self._image_files = []

        self.dicom_directory = os.path.join(self._input_folder, DICOM_FOLDER)
        if not os.path.isdir(self.dicom_directory):
            raise ValueError(f"No dicom folder found, it must be named {DICOM_FOLDER}")

        self.jpg_directory = os.path.join(self._input_folder, JPG_FOLDER)
        if not os.path.isdir(self.jpg_directory):
            raise ValueError(f"No jpg folder found, it must be named {JPG_FOLDER}")

        self.jpg_name_to_index, self.index_to_jpg_name = get_maps(self.jpg_directory)

        self.all_image_data = parse_csv(
            self.csv_file, self.jpg_directory, self.index_to_jpg_name
        )

    def set_input_folder(self, folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            raise ValueError("Invalid folder path provided.")
        self._input_folder = folder_path

    def save_image_to_output_folder(self, image, image_index):
        output_path = f"{self._output_folder}/{self.get_output_image_name(image_index)}"
        cv2.imwrite(output_path, image)

    def get_dicom_image(self, image_index):
        if len(self._image_files) <= 0:
            raise ValueError("No images found in the folder.")
        if image_index < 0 or image_index >= len(self._image_files):
            raise IndexError("Image index out of range.")

        image_path = os.path.join(self._input_folder, self._image_files[image_index])
        return pydicom.dcmread(image_path).pixel_array

    def read_jpg_image(self, image_path):
        return cv2.imread(image_path)

    def get_dicom_crop(self, index):
        image_data = self.all_image_data[index]
        jpg_image_path = image_data.image_path
        dicom_image_path = get_corresponding_dicom_image(
            jpg_image_path, self.dicom_directory, self.jpg_name_to_index
        )

        cx = image_data.cx
        cy = image_data.cy
        rx = image_data.rx
        ry = image_data.ry

        image = pydicom.dcmread(dicom_image_path).pixel_array
        ratio = get_ratio(self.jpg_directory, self.dicom_directory)
        x_start = int((cx - rx + ratio - 1) // ratio)
        x_end = int((cx + rx + ratio - 1) // ratio)
        y_start = int((cy - ry + ratio - 1) // ratio)
        y_end = int((cy + ry + ratio - 1) // ratio)
        image = image[y_start:y_end, x_start:x_end]

        return image, jpg_image_path, image_data

    def set_output_folder(self, folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            raise ValueError("No folder provided. Maybe you selected a file instead.")

        if len(os.listdir(folder_path)) > 0:
            raise ValueError(
                "Output folder is not empty. Select an empty folder to avoid overriding sensitive data."
            )

        self._output_folder = folder_path

    def get_processed_image(self, image_index):
        if self._output_folder is None:
            raise ValueError("Output folder is not set. ")

        output_folder_length = len(os.listdir(self._output_folder))

        if output_folder_length == 0:
            raise ValueError("No images found in the output folder.")
        if image_index < 0 or image_index >= output_folder_length:
            raise IndexError("Image index out of range.")

        image_path = os.path.join(
            self._output_folder, self.get_output_image_name(image_index)
        )
        return cv2.imread(image_path)

    def crop_count(self):
        return len(self.all_image_data)

    def output_folder_length(self):
        return len(os.listdir(self._output_folder))

    def image_count(self):
        return len(self._image_files)
