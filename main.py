import sys
import os

# import torch
# from PIL import Image
# import torchvision.transforms as T
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt

# # Load the trained DETR model
# model = torch.jit.load("detr_scripted.pt")  # Ensure the model is in the same folder
# model.eval()

# # Function to preprocess the image
# def preprocess_image(image_path):
#     transform = T.Compose([
#         T.Resize((800, 800)),
#         T.ToTensor(),
#     ])
#     image = Image.open(image_path).convert("RGB")
#     return transform(image).unsqueeze(0)  # Add batch dimension

# # Function to run inference
# def detect_objects(image_path):
#     image_tensor = preprocess_image(image_path)
#     with torch.no_grad():
#         outputs = model(image_tensor)
#     return outputs  # You may want to format this properly


# GUI Application
class DetrApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_folder = None
        self.image_files = []
        self.current_index = 0

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
        self.image_label.setFixedSize(500, 350)  # Default size

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
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            self.image_files.sort()  # Sort alphabetically

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
            QApplication.processEvents()  # Update UI immediately

            # Load and display the image
            pixmap = QPixmap(
                file_path
            )  # TODO: make it work with dicoms, maybe use pydicom and aget the pixelarray
            self.image_label.setPixmap(
                pixmap.scaled(500, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # Run object detection
            results = "here"  # detect_objects(file_path)

            self.status_label.setText(
                f"Detection complete! ({self.current_index + 1}/{len(self.image_files)})"
            )
            print(
                "Detection Results:", results
            )  # You can display results in the UI later

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


# Run the App
app = QApplication(sys.argv)
window = DetrApp()
window.show()
sys.exit(app.exec())
