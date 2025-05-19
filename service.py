import numpy as np
import cv2

import torch
from torchvision.models import efficientnet_b0
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2

from repository import Repository
from utils.subject import Subject
from utils.classification.prediction_data import PredictionData
from utils.project_config import (
    DEVICE,
    T2_CLASSIFICATION_MODEL,
    T2_DETECTION_MODEL,
    T2_IMAGE,
    T2_IOU_THRESHOLD,
    T2_DETECTION_IMAGE_MEAN,
    T2_DETECTION_IMAGE_STD,
    DWI_CLASSIFICATION_MODEL,
    DWI_DETECTION_MODEL,
    DWI_IMAGE,
    DWI_IOU_THRESHOLD,
    DWI_DETECTION_IMAGE_MEAN,
    DWI_DETECTION_IMAGE_STD,
    DETECTION_TASK,
    CLASSIFICATION_TASK,
)


class Service(Subject):
    def __init__(self, repository: Repository):
        super().__init__()
        self._repository = repository
        self._model_type = None
        self._min_iou_threshold = None
        self.classification_predictions = {}

    def load_t2_detection_model(self):
        model = fcos_resnet50_fpn(trainable_backbone_layers=5)
        model.transform = GeneralizedRCNNTransform(
            min_size=(256,),
            max_size=256,
            image_mean=T2_DETECTION_IMAGE_MEAN,
            image_std=T2_DETECTION_IMAGE_STD,
        )
        model.load_state_dict(torch.load(T2_DETECTION_MODEL, map_location=DEVICE))
        model.eval()

        self._model_type = T2_IMAGE
        self._min_iou_threshold = T2_IOU_THRESHOLD
        return model

    def load_dwi_detection_model(self):
        model = fcos_resnet50_fpn(trainable_backbone_layers=5)
        model.transform = GeneralizedRCNNTransform(
            min_size=(1024,),
            max_size=1333,
            image_mean=DWI_DETECTION_IMAGE_MEAN,
            image_std=DWI_DETECTION_IMAGE_STD,
        )
        model.load_state_dict(torch.load(DWI_DETECTION_MODEL, map_location=DEVICE))
        model.eval()

        self._model_type = DWI_IMAGE
        self._min_iou_threshold = DWI_IOU_THRESHOLD
        return model

    def load_classification_model(self, image_type):
        if image_type == T2_IMAGE:
            model_path = T2_CLASSIFICATION_MODEL
        elif image_type == DWI_IMAGE:
            model_path = DWI_CLASSIFICATION_MODEL
        else:
            raise ValueError(f"The provided image type is not valid.")

        model = efficientnet_b0()
        model.features[0][0] = torch.nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=2, bias=True),
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model

    def set_image_folder(self, folder_path):
        self._repository.set_input_folder(folder_path)

    def output_count(self):
        return self._repository.output_folder_length()

    def _plot_detection_predictions(self, image, predictions):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for box, label, score in zip(
            predictions["boxes"], predictions["labels"], predictions["scores"]
        ):
            score = float(score)
            if self._model_type == T2_IMAGE:
                min_score = 0.2
            elif self._model_type == DWI_IMAGE:
                min_score = 0.5

            if score < min_score:
                continue

            x1, y1, x2, y2 = map(int, box)
            label = str(label)
            color = (255, 0, 0)

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

        print("finished plotting predictions")
        return image

    def _plot_and_save_all_classification_predictions(self):
        for i, image_path in enumerate(self.classification_predictions.keys()):
            image = self._repository.read_jpg_image(image_path)
            for prediction_data in self.classification_predictions[image_path]:
                if (
                    prediction_data.predicted_class == "malignant"
                ):  # TODO: get rid of the magic string
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                top_left_corner = (
                    int(prediction_data.top_left_corner[0]),
                    int(prediction_data.top_left_corner[1]),
                )
                bottom_right_corner = (
                    int(prediction_data.bottom_right_corner[0]),
                    int(prediction_data.bottom_right_corner[1]),
                )

                # print(type(image))
                image = cv2.rectangle(
                    image, top_left_corner, bottom_right_corner, color, 1
                )
                image = cv2.putText(
                    image,
                    f"{prediction_data.predicted_class}: {prediction_data.confidence:.2f}%",
                    (top_left_corner[0], top_left_corner[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.3,
                    color,
                )
            self._repository.save_image_to_output_folder(image, i)

    def process_classification_image(self, model, image_index):
        cropped_image, jpg_image_path, image_data = self._repository.get_dicom_crop(
            image_index
        )
        transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.ToDtype(torch.float32),
            ]
        )
        cropped_image = transform(cropped_image)
        cropped_image = torch.unsqueeze(cropped_image, 0)
        cropped_image = cropped_image.to(DEVICE)
        logits = model(cropped_image)
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probs)
        confidence = torch.max(probs).item() * 100
        if prediction == 0:  # TODO: get rid of the magic number and strings
            predicted_class = "benign"
        else:
            predicted_class = "malignant"
        top_left_corner = (image_data.cx - image_data.rx, image_data.cy - image_data.ry)
        bottom_right_corner = (
            image_data.cx + image_data.rx,
            image_data.cy + image_data.ry,
        )
        prediction_data = PredictionData(
            predicted_class, confidence, top_left_corner, bottom_right_corner
        )

        if jpg_image_path not in self.classification_predictions.keys():
            self.classification_predictions[jpg_image_path] = []
        self.classification_predictions[jpg_image_path].append(prediction_data)

    def process_detection_image(self, model, image_index):
        image = self._repository.get_dicom_image(image_index)

        # Prepare the inference image by bringing it to [0, 1],
        # replicating it across the 3 channels and applying the transforms
        inference_image = torch.from_numpy(image).float()
        if inference_image.min() == inference_image.max():
            raise ValueError(
                "All pixels values are the same. This is not the correct image, or maybe it was corruted."
            )
        inference_image = (inference_image - inference_image.min()) / (
            inference_image.max() - inference_image.min()
        )
        inference_image = inference_image.unsqueeze(0)
        inference_image = inference_image.repeat(3, 1, 1)

        transform = v2.Compose(
            [v2.Resize(size=(256, 256)), v2.ToDtype(torch.float32, scale=False)]
        )
        inference_image = transform(inference_image)

        # Get predictions and plot them on the original image
        predictions = model([inference_image])[0]
        plotted_image = self._plot_detection_predictions(image, predictions)

        return plotted_image

    def load_model(self, image_type, task_type):
        if image_type == T2_IMAGE and task_type == DETECTION_TASK:
            return self.load_t2_detection_model()
        elif image_type == DWI_IMAGE and task_type == DETECTION_TASK:
            return self.load_dwi_detection_model()
        elif task_type == CLASSIFICATION_TASK:
            return self.load_classification_model(image_type)
        else:
            raise ValueError(f"Invalid combination of image type and task type.")

    def process_all_images(self, image_type, task_type):
        if self._repository.output_folder_is_set() is False:
            raise ValueError("Output folder is not set. Please set it first.")

        model = self.load_model(image_type, task_type)

        if task_type == DETECTION_TASK:
            self._repository.prepare_input_files_for_detection()
            for i in range(self._repository.image_count()):
                image = self.process_detection_image(model, i)
                self._repository.save_image_to_output_folder(image, i)
                self.notify_observers(
                    processed_image_count=i + 1,
                    total_image_count=self._repository.image_count(),
                )
        elif task_type == CLASSIFICATION_TASK:
            self.classification_predictions = {}
            self._repository.prepare_input_files_for_classification()
            for i in range(self._repository.crop_count()):
                self.process_classification_image(model, i)
                self.notify_observers(
                    processed_image_count=i + 1,
                    total_image_count=self._repository.crop_count(),
                )
            self._plot_and_save_all_classification_predictions()

    def set_output_folder(self, folder_path):
        self._repository.set_output_folder(folder_path)

    def get_processed_image(self, image_index):
        return self._repository.get_processed_image(image_index)
