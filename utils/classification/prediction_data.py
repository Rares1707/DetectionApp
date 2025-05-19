class PredictionData:
    def __init__(
        self, predicted_class, confidence, top_left_corner, bottom_right_corner
    ):
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.top_left_corner = top_left_corner
        self.bottom_right_corner = bottom_right_corner
