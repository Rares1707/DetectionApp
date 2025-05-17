import sys

from PySide6.QtWidgets import QApplication

from repository import Repository
from service import Service
from gui import MainWindow

if __name__ == "__main__":
    repository = Repository()
    service = Service(repository)

    app = QApplication(sys.argv)
    window = MainWindow(service=service)
    service.register_observer(window)
    window.show()
    sys.exit(app.exec())
