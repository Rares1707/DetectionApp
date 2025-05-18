from functools import wraps

from PySide6.QtWidgets import QMessageBox


def catch_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, IndexError, NotImplementedError) as exception:
            message_box = QMessageBox()
            message_box.setText(str(exception))
            message_box.exec()

    return wrapper
