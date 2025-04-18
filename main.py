import sys
import os
from torchvision.transforms import v2
from PIL import Image
import torch
import pydicom
import cv2
import numpy as np
import pickle
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)
from PySide6.QtGui import QPixmap, QFont, QImage
from PySide6.QtCore import Qt


class DetrApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_folder = None
        self.image_files = []
        self.current_index = 0
        self.model = model = fcos_resnet50_fpn(trainable_backbone_layers=5)
        model.transform = GeneralizedRCNNTransform(
            min_size=(256,),
            max_size=256,
            image_mean=[0.2341376394033432, 0.2341376394033432, 0.2341376394033432],
            image_std=[0.2010965347290039, 0.2010965347290039, 0.2010965347290039],
        )
        self.model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        self.model.eval()
        print("model loaded")

    def initUI(self):
        self.setWindowTitle("DETR Object Detection")
        self.setGeometry(200, 100, 700, 500)
        self.setStyleSheet(
            """
            QWidget { background-color: #2c3e50; color: white; font-size: 14px; }
            QPushButton { background-color: #3498db; color: white; border-radius: 5px; padding: 10px; font-size: 16px; }
            QPushButton:hover { background-color: #2980b9; }
            QLabel { font-size: 16px; }
        """
        )

        # Title Label
        self.title = QLabel("DETR Object Detection", self)
        self.title.setFont(QFont("Arial", 18, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)

        # Image Preview
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed white; padding: 10px;")
        self.image_label.setFixedSize(512, 512)  # Default size

        # Status Label
        self.status_label = QLabel("Select a folder to detect objects", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.select_folder_btn = QPushButton("Choose Folder", self)
        self.select_folder_btn.clicked.connect(self.load_folder)

        self.prev_btn = QPushButton("Previous", self)
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.select_folder_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.image_folder = folder_path
            self.image_files = [
                f for f in os.listdir(folder_path) if f.lower().endswith((".dcm"))
            ]
            self.image_files.sort()  # sort alphabetically

            if self.image_files:
                self.current_index = 0
                self.show_image()
                self.next_btn.setEnabled(len(self.image_files) > 1)
                self.prev_btn.setEnabled(False)

    def show_image(self):
        if self.image_folder and self.image_files:
            file_path = os.path.join(
                self.image_folder, self.image_files[self.current_index]
            )
            self.status_label.setText(
                f"Processing: {self.image_files[self.current_index]}"
            )
            QApplication.processEvents()

            # Load the DICOM image
            dicom_data = pydicom.dcmread(file_path)
            image = dicom_data.pixel_array
            inference_image = torch.from_numpy(image).float()
            if inference_image.min() == inference_image.max():
                raise
            inference_image = (inference_image - inference_image.min()) / (
                inference_image.max() - inference_image.min()
            )
            inference_image = inference_image.unsqueeze(0)
            inference_image = inference_image.repeat(3, 1, 1)

            transform = v2.Compose(
                [v2.Resize(size=(256, 256)), v2.ToDtype(torch.float32, scale=False)]
            )
            inference_image = transform(inference_image)

            predictions = self.model([inference_image])[0]
            # print("results predicted")

            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # print(predictions)
            for box, label, score in zip(
                predictions["boxes"], predictions["labels"], predictions["scores"]
            ):
                score = float(score)
                if score >= 0.2:
                    color = (0, 255, 0)
                    if label == 2:
                        color = (255, 0, 0)

                    x1, y1, x2, y2 = map(int, box)
                    label = str(label)

                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

                    # cv2.putText(
                    #     image,
                    #     f"{score:.2f}",
                    #     (x1, y1 - 5),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.2,
                    #     color,
                    #     1,
                    # )

            print("finished drawing boxes")

            # Convert the image back to QPixmap for display
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QPixmap.fromImage(
                QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            )
            self.image_label.setPixmap(
                q_image.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            self.status_label.setText(
                f"Detection complete! ({self.current_index + 1}/{len(self.image_files)})"
            )

    def show_next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()
            self.prev_btn.setEnabled(True)
            if self.current_index == len(self.image_files) - 1:
                self.next_btn.setEnabled(False)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
            self.next_btn.setEnabled(True)
            if self.current_index == 0:
                self.prev_btn.setEnabled(False)


app = QApplication(sys.argv)
window = DetrApp()
window.show()
sys.exit(app.exec())
