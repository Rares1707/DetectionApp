from .patient import Patient


class ImageData:
    def __init__(
        self,
        original_image_path,
        label,
        patient: Patient,
        region_type,
        cx,
        cy,
        rx,
        ry=None,
        theta=None,
    ):
        self.image_path = original_image_path
        self.label = label
        self.region_type = region_type
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        if region_type == "circle":
            self.ry = rx
        self.theta = theta
        self.patient = patient
