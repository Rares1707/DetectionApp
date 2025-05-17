import os

import pydicom


class Repository:
    def __init__(self):
        self.output_folder = None
        self.image_folder = None
        self.csv_file = None
        self.image_files = []

    def set_image_folder(self, folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            raise ValueError("Invalid folder path provided.")

        self.image_folder = folder_path
        self.image_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith((".dcm"))
        ]

        if len(self.image_files) == 0:
            raise ValueError("No DICOM files found in the folder.")

        self.image_files.sort()

    def get_image_path(self, image_index):
        if len(self.image_files) <= 0:
            raise ValueError("No images found in the folder.")

        if image_index < 0 or image_index >= len(self.image_files):
            raise IndexError("Image index out of range.")

        return os.path.join(self.image_folder, self.image_files[image_index])

    def get_image(self, image_index):
        image_path = self.get_image_path(image_index)
        return pydicom.dcmread(image_path).pixel_array

    def set_output_folder(self, folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            raise ValueError("No folder provided. Maybe you selected a file instead.")
        self.output_folder = folder_path

    def __len__(self):
        return len(self.image_files)
