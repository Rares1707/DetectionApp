from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QRadioButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)
from PySide6.QtGui import QPixmap, QFont, QImage
from PySide6.QtCore import Qt

from service import Service


class DetrApp(QWidget):
    def __init__(self, service: Service):
        super().__init__()
        self._service = service
        self._current_index = 0
        self.initUI()
        # self.load_t2_model()

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

        self.title = QLabel("DETR Object Detection", self)
        self.title.setFont(QFont("Arial", 18, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed white; padding: 10px;")
        self.image_label.setFixedSize(512, 512)

        self.status_label = QLabel("Select a folder to detect objects", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.image_type_label = QLabel("Choose image type", self)
        self.image_type_label.setAlignment(Qt.AlignCenter)

        self.select_output_folder_btn = QPushButton("Choose Output Folder", self)
        self.select_output_folder_btn.clicked.connect(self.choose_output_folder)

        self.select_input_folder_btn = QPushButton("Choose Input Folder", self)
        self.select_input_folder_btn.clicked.connect(self.load_folder)

        self.prev_btn = QPushButton("Previous", self)
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)

        self.t2_radio_btn = QRadioButton("T2", self)
        self.t2_radio_btn.toggled.connect(self.load_t2_model)
        self.t2_radio_btn.setChecked(True)

        self.dwi_radio_btn = QRadioButton("DWI", self)
        self.dwi_radio_btn.toggled.connect(self.load_dwi_model)
        self.dwi_radio_btn.setChecked(False)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_type_label)
        layout.addWidget(self.t2_radio_btn)
        layout.addWidget(self.dwi_radio_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.select_output_folder_btn)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.select_input_folder_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    def load_t2_model(self):
        if not self.t2_radio_btn.isChecked():
            return
        self._service.load_t2_model()

    def load_dwi_model(self):
        if not self.dwi_radio_btn.isChecked():
            return
        self._service.load_dwi_model()

    def choose_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        self._service.set_output_folder(folder_path)
        self.status_label.setText(f"Output folder set to: {folder_path}")

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        self._service.set_image_folder(folder_path)
        self._service.process_all_images()

        # TODO:
        # self.service.process_all_images() # this should also save the images to the output folder
        # modify show_image to just load the image from the output folder (don't forget to keep the layered architecture principle)

        self._current_index = 0
        self.show_image()
        self.next_btn.setEnabled(self._service.get_image_count() > 1)
        self.prev_btn.setEnabled(False)

    def show_image(self):
        self.status_label.setText("Processing image...")
        QApplication.processEvents()

        # image = self._service.process_image(self._current_index)
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
            f"Detection complete! ({self._current_index + 1}/{self._service.get_image_count()})"
        )

    def show_next_image(self):
        if self._current_index < self._service.get_image_count() - 1:
            self._current_index += 1
            self.show_image()
            self.prev_btn.setEnabled(True)
            if self._current_index == self._service.get_image_count() - 1:
                self.next_btn.setEnabled(False)

    def show_previous_image(self):
        if self._current_index > 0:
            self._current_index -= 1
            self.show_image()
            self.next_btn.setEnabled(True)
            if self._current_index == 0:
                self.prev_btn.setEnabled(False)
