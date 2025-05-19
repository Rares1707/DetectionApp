import sys

from PySide6.QtWidgets import QApplication

from repository import Repository
from service import Service
from gui import MainWindow

# TODO: switch to GPU if available (let's see how you test it though, maybe use another machine?)
# TODO: change many public fields/methods to private
if __name__ == "__main__":
    repository = Repository()
    service = Service(repository)

    app = QApplication(sys.argv)
    window = MainWindow(service=service)
    service.register_observer(window)
    window.show()
    sys.exit(app.exec())
