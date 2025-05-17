import sys

from PySide6.QtWidgets import QApplication

from repository import Repository
from service import Service
from gui import DetrApp


# TODO: store images in the output directory
# TODO: maybe just process all images and put them in the output directory and THEN iterate through them for visualization.
# TODO: Why? Because now if you press "previous" it will have to process the image again.
if __name__ == "__main__":
    repository = Repository()
    service = Service(repository)

    app = QApplication(sys.argv)
    window = DetrApp(service=service)
    window.show()
    sys.exit(app.exec())
