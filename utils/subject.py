from .observer import Observer


class Subject:
    def __init__(self):
        self._observers = []

    def register_observer(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, **kwargs):
        for observer in self._observers:
            observer.refresh(**kwargs)
