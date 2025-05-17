import numpy as np
import cv2

import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2

from repository import Repository
from project_constants import (
    T2_DETECTION_MODEL,
    T2_MODEL_TYPE,
    T2_IOU_THRESHOLD,
    T2_IMAGE_MEAN,
    T2_IMAGE_STD,
    DWI_DETECTION_MODEL,
    DWI_MODEL_TYPE,
    DWI_IOU_THRESHOLD,
    DWI_IMAGE_MEAN,
    DWI_IMAGE_STD,
)


class Service:
    def __init__(self, repository: Repository):
        self._repository = repository
        self._model = None
        self._model_type = None
        self._min_iou_threshold = None

    def load_t2_model(self):
        self._model = fcos_resnet50_fpn(trainable_backbone_layers=5)
        self._model.transform = GeneralizedRCNNTransform(
            min_size=(256,),
            max_size=256,
            image_mean=T2_IMAGE_MEAN,
            image_std=T2_IMAGE_STD,
        )
        self._model.load_state_dict(torch.load(T2_DETECTION_MODEL, map_location="cpu"))
        self._model.eval()
        self._model_type = T2_MODEL_TYPE
        self._min_iou_threshold = T2_IOU_THRESHOLD
        print("T2 model loaded")

    def load_dwi_model(self):
        self._model = fcos_resnet50_fpn(trainable_backbone_layers=5)
        self._model.transform = GeneralizedRCNNTransform(
            min_size=(1024,),
            max_size=1333,
            image_mean=DWI_IMAGE_MEAN,
            image_std=DWI_IMAGE_STD,
        )
        self._model.load_state_dict(torch.load(DWI_DETECTION_MODEL, map_location="cpu"))
        self._model.eval()
        self._model_type = DWI_MODEL_TYPE
        self._min_iou_threshold = DWI_IOU_THRESHOLD
        print("DWI model loaded")

    def set_image_folder(self, folder_path):
        self._repository.set_image_folder(folder_path)

    def get_image_count(self):
        return len(self._repository)

    def get_image_path(self, image_index):
        return self._repository.get_image_path(image_index)

    def _plot_predictions(self, image, predictions):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for box, label, score in zip(
            predictions["boxes"], predictions["labels"], predictions["scores"]
        ):
            score = float(score)
            if self._model_type == T2_MODEL_TYPE:
                min_score = 0.2
            elif self._model_type == DWI_MODEL_TYPE:
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

    def process_image(self, image_index):
        image = self._repository.get_image(image_index)

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
        predictions = self._model([inference_image])[0]
        plotted_image = self._plot_predictions(image, predictions)

        return plotted_image

    def process_all_images(self):
        if self._repository.output_folder_is_set() is False:
            raise ValueError("Output folder is not set. Please set it first.")

        for i in range(len(self._repository)):
            image = self.process_image(i)
            self._repository.save_image_to_output_folder(image, i)

    def set_output_folder(self, folder_path):
        self._repository.set_output_folder(folder_path)

    def get_processed_image(self, image_index):
        return self._repository.get_processed_image(image_index)
