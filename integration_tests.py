import os
import unittest
import shutil
from repository import Repository
from service import Service
from utils.project_config import (
    DETECTION_TASK,
    CLASSIFICATION_TASK,
    T2_IMAGE,
    DWI_IMAGE,
)


class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Repository()
        cls.service = Service(cls.repository)
        cls.t2_classification_input_folder_name = "t2_classification_sample"
        cls.t2_detection_input_folder_name = "t2_detection_sample"
        cls.dwi_classification_input_folder_name = "dwi_classification_sample"
        cls.dwi_detection_input_folder_name = "dwi_detection_sample"

        cls.output_folder_name = "test_output_folder"
        os.mkdir(cls.output_folder_name)
        cls.service.set_output_folder(cls.output_folder_name)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.output_folder_name)

    def empty_output_folder(self):
        for file_name in os.listdir(self.output_folder_name):
            file_path = os.path.join(self.output_folder_name, file_name)
            os.remove(file_path)

    def test_service_t2_classification(self):
        self.service.set_image_folder(self.t2_classification_input_folder_name)
        self.service.process_all_images(T2_IMAGE, CLASSIFICATION_TASK)
        output_folder_length = len(os.listdir(self.output_folder_name))
        self.assertGreater(output_folder_length, 0)
        self.empty_output_folder()

    def test_service_t2_object_detection(self):
        self.service.set_image_folder(self.t2_detection_input_folder_name)
        self.service.process_all_images(T2_IMAGE, DETECTION_TASK)
        output_folder_length = len(os.listdir(self.output_folder_name))
        self.assertGreater(output_folder_length, 0)
        self.empty_output_folder()

    def test_service_dwi_classification(self):
        self.service.set_image_folder(self.dwi_classification_input_folder_name)
        self.service.process_all_images(DWI_IMAGE, CLASSIFICATION_TASK)
        output_folder_length = len(os.listdir(self.output_folder_name))
        self.assertGreater(output_folder_length, 0)
        self.empty_output_folder()

    def test_service_dwi_object_detection(self):
        self.service.set_image_folder(self.dwi_detection_input_folder_name)
        self.service.process_all_images(DWI_IMAGE, DETECTION_TASK)
        output_folder_length = len(os.listdir(self.output_folder_name))
        self.assertGreater(output_folder_length, 0)
        self.empty_output_folder()


if __name__ == "__main__":
    unittest.main()
