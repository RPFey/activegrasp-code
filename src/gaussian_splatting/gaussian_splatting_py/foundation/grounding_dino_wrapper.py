import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.draw.color import ColorPalette
import tyro
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GroundingDINO:
    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    def __init__(self, grounding_model="IDEA-Research/grounding-dino-base"):
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_id = grounding_model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    def __call__(self, image: Image, text: str):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results


def test_grounding_dino(img_path: str,
                        /,
                        text: str="car. tire.",):
    CUSTOM_COLOR_MAP = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#0082c8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#d2f53c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#aa6e28",
        "#fffac8",
        "#800000",
        "#aaffc3",
    ]
    grounding_dino = GroundingDINO()
    image = Image.open(img_path)
    results = grounding_dino(image, text)

    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    input_boxes = results[0]["boxes"].cpu().numpy()
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        class_id=class_ids
    )

    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join("groundingdino_annotated_image.jpg"), annotated_frame)
    print(results)
    cv2.imshow("image", annotated_frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    tyro.cli(test_grounding_dino)
    