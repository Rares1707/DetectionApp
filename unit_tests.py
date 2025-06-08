import os
import unittest
from repository import Repository


class TestRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Repository()
        cls.output_folder_name = "mock_output_folder"
        cls.input_folder_name = "mock_input_folder"
        os.mkdir(cls.output_folder_name)
        os.mkdir(cls.input_folder_name)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.output_folder_name)
        os.rmdir(cls.input_folder_name)

    def test_valid_csv(self):
        self.repository.csv_file = r"utils\test_files\valid.csv"
        csv_is_valid = self.repository.csv_is_valid()
        self.assertTrue(csv_is_valid)

    def test_invalid_csv(self):
        self.repository.csv_file = r"utils\test_files\invalid.csv"
        csv_is_valid = self.repository.csv_is_valid()
        self.assertFalse(csv_is_valid)

    def test_empty_csv(self):
        self.repository.csv_file = r"utils\test_files\empty.csv"
        csv_is_valid = self.repository.csv_is_valid()
        self.assertFalse(csv_is_valid)

    def test_set_output_folder(self):
        self.repository.set_output_folder(self.output_folder_name)
        self.assertEqual(self.repository._output_folder, self.output_folder_name)

    def test_set_output_folder_none(self):
        try:
            self.repository.set_output_folder(None)
            self.fail()
        except ValueError:
            self.assertTrue(True)
        except Exception:
            self.fail("Expected Value error but got a different error.")

    def test_output_folder_is_set(self):
        self.repository._output_folder = "mock_output_folder"
        self.assertTrue(self.repository.output_folder_is_set())

    def test_output_folder_is_not_set(self):
        self.assertFalse(self.repository.output_folder_is_set())


if __name__ == "__main__":
    unittest.main()
