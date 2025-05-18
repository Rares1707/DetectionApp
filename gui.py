from typing import override
from functools import wraps

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QRadioButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMessageBox,
    QButtonGroup,
)
from PySide6.QtGui import QPixmap, QFont, QImage
from PySide6.QtCore import Qt

from service import Service
from utils.observer import Observer
from utils.catch_exceptions_decorator import catch_exceptions
from utils.project_constants import (
    DETECTION_TASK,
    CLASSIFICATION_TASK,
    T2_IMAGE,
    DWI_IMAGE,
)


class MainWindow(QWidget, Observer):
    def __init__(self, service: Service):
        super().__init__()
        self._service = service
        self._current_index = 0
        self._message_box = QMessageBox()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Cervical Cancer Detection")
        self.setGeometry(200, 100, 700, 500)
        self.setStyleSheet(
            """
            QWidget { background-color: #2c3e50; color: white; font-size: 14px; }
            QPushButton { background-color: #3498db; color: white; border-radius: 5px; padding: 10px; font-size: 16px; }
            QPushButton:hover { background-color: #2980b9; }
            QLabel { font-size: 16px; }
        """
        )

        self.title = QLabel("Cervical Cancer Detection", self)
        self.title.setFont(QFont("Arial", 18, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed white; padding: 10px;")
        self.image_label.setFixedSize(512, 512)

        self.status_label = QLabel("Select an output folder", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.image_type_label = QLabel("Choose image type", self)
        self.image_type_label.setAlignment(Qt.AlignLeft)

        self.task_type_label = QLabel("Choose task type", self)
        self.task_type_label.setAlignment(Qt.AlignLeft)

        self.select_output_folder_btn = QPushButton("Choose Output Folder", self)
        self.select_output_folder_btn.clicked.connect(self.choose_output_folder)

        self.select_input_folder_btn = QPushButton("Choose Input Folder", self)
        self.select_input_folder_btn.clicked.connect(self.choose_input_folder)

        self.prev_btn = QPushButton("Previous", self)
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)

        self.t2_radio_btn = QRadioButton("T2", self)
        self.t2_radio_btn.setChecked(False)
        self.dwi_radio_btn = QRadioButton("DWI", self)
        self.dwi_radio_btn.setChecked(False)

        self.image_type_group = QButtonGroup(self)
        self.image_type_group.addButton(self.t2_radio_btn)
        self.image_type_group.addButton(self.dwi_radio_btn)

        self.classification_radio_btn = QRadioButton("Classification", self)
        self.classification_radio_btn.setChecked(False)
        self.object_detection_radio_btn = QRadioButton("Object Detection", self)
        self.object_detection_radio_btn.setChecked(False)

        self.task_type_group = QButtonGroup(self)
        self.task_type_group.addButton(self.classification_radio_btn)
        self.task_type_group.addButton(self.object_detection_radio_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.image_label)

        image_type_layout = QVBoxLayout()
        image_type_layout.addWidget(self.image_type_label)
        # image_type_layout.addWidget(self.image_type_group)
        image_type_layout.addWidget(self.t2_radio_btn)
        image_type_layout.addWidget(self.dwi_radio_btn)

        task_type_layout = QVBoxLayout()
        task_type_layout.addWidget(self.task_type_label)
        # task_type_layout.addWidget(self.task_type_group)
        task_type_layout.addWidget(self.classification_radio_btn)
        task_type_layout.addWidget(self.object_detection_radio_btn)

        radio_buttons_layout = QHBoxLayout()
        radio_buttons_layout.addLayout(image_type_layout)
        radio_buttons_layout.addLayout(task_type_layout)

        layout.addLayout(radio_buttons_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.select_output_folder_btn)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.select_input_folder_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    @override
    def refresh(self, **kwargs):
        processed_images_count = kwargs.get("processed_image_count")
        total_image_count = kwargs.get("total_image_count")

        if processed_images_count is None or total_image_count is None:
            return

        self.status_label.setText(
            f"Processing images... ({processed_images_count}/{total_image_count})"
        )
        QApplication.processEvents()

    # def load_t2_model(self):
    #     if not self.t2_radio_btn.isChecked():
    #         return
    #     self._service.load_t2_model()

    # def load_dwi_model(self):
    #     if not self.dwi_radio_btn.isChecked():
    #         return
    #     self._service.load_dwi_model()

    @catch_exceptions
    def choose_output_folder(self):
        try:
            folder_path = QFileDialog.getExistingDirectory(self, "Select output folder")
            self._service.set_output_folder(folder_path)
            self.status_label.setText(f"Output folder set. Now select an input folder.")
        except ValueError as exception:
            self._message_box.setText(str(exception))
            self._message_box.exec()

    @catch_exceptions
    def choose_input_folder(self):
        """ "
        Careful, this method also starts the inference process.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select input folder")
        self._service.set_image_folder(folder_path)

        self.process_all_images()

    def process_all_images(self):
        if self.t2_radio_btn.isChecked():
            image_type = T2_IMAGE
        elif self.dwi_radio_btn.isChecked():
            image_type = DWI_IMAGE
        else:
            message_box = QMessageBox()
            message_box.setText("You forgot to select the image type.")
            message_box.exec()
            return

        if self.classification_radio_btn.isChecked():
            task_type = CLASSIFICATION_TASK
        elif self.object_detection_radio_btn.isChecked():
            task_type = DETECTION_TASK
        else:
            message_box = QMessageBox()
            message_box.setText("You forgot to select the task type.")
            message_box.exec()
            return

        self._service.process_all_images(image_type, task_type)

        self._current_index = 0
        self.show_image()
        self.next_btn.setEnabled(self._service.get_image_count() > 1)
        self.prev_btn.setEnabled(False)

    @catch_exceptions
    def show_image(self):
        image = self._service.get_processed_image(self._current_index)

        # Convert the image to QPixmap
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QPixmap.fromImage(
            QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        )
        self.image_label.setPixmap(
            q_image.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        self.status_label.setText(
            f"Done! Showing image {self._current_index + 1}/{self._service.get_image_count()}"
        )

    def show_next_image(self):
        if self._current_index >= self._service.get_image_count() - 1:
            return
        self._current_index += 1
        self.show_image()
        self.prev_btn.setEnabled(True)
        if self._current_index == self._service.get_image_count() - 1:
            self.next_btn.setEnabled(False)

    def show_previous_image(self):
        if self._current_index <= 0:
            return
        self._current_index -= 1
        self.show_image()
        self.next_btn.setEnabled(True)
        if self._current_index == 0:
            self.prev_btn.setEnabled(False)
