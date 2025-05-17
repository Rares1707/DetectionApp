import os

import cv2
import pydicom


class Repository:
    def __init__(self):
        self._output_folder = None
        self._image_folder = None
        self._csv_file = None
        self._image_files = []

    def output_folder_is_set(self):
        return self._output_folder is not None

    def set_image_folder(self, folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            raise ValueError("Invalid folder path provided.")

        self._image_folder = folder_path
        self._image_files = [
            file_name
            for file_name in os.listdir(folder_path)
            if file_name.lower().endswith((".dcm"))
        ]

        if len(self._image_files) == 0:
            raise ValueError("No DICOM files found in the folder.")

        self._image_files.sort()

    def save_image_to_output_folder(self, image, image_index):
        output_path = f"{self._output_folder}/{image_index}.png"
        cv2.imwrite(output_path, image)

    def get_image_path(self, image_index):
        if len(self._image_files) <= 0:
            raise ValueError("No images found in the folder.")

        if image_index < 0 or image_index >= len(self._image_files):
            raise IndexError("Image index out of range.")

        return os.path.join(self._image_folder, self._image_files[image_index])

    def get_image(self, image_index):
        image_path = self.get_image_path(image_index)
        return pydicom.dcmread(image_path).pixel_array

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

        image_path = os.path.join(self._output_folder, f"{image_index}.png")
        return cv2.imread(image_path)

    def __len__(self):
        return len(self._image_files)
