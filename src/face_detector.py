"""
Face Detection Module supporting both Haar Cascade and MTCNN.
Fulfills Step 4: Use face detection models (Haar Cascade / MTCNN) to locate faces.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

import config

Union_Img = Union[np.ndarray, Image.Image]


class HaarCascadeDetector:
    """Face detector using OpenCV's Haar Feature-based Cascade Classifiers."""

    def __init__(self, scale_factor: float = config.HAAR_SCALE_FACTOR, min_neighbors: int = config.HAAR_MIN_NEIGHBORS):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image: Union_Img) -> List[Dict[str, Any]]:
        """
        Detects faces in RGB numpy array or PIL image.
        Returns list of dicts: [{'box': (x, y, w, h), 'confidence': float, 'face_crop': PIL.Image, 'detector': 'haar'}]
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            rgb_arr = np.array(image.convert("RGB"))
        else:
            rgb_arr = image.copy()
            if len(rgb_arr.shape) == 2:
                rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2RGB)

        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        # Haar detectMultiScale
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=config.HAAR_MIN_SIZE
        )

        results = []
        h_img, w_img = rgb_arr.shape[:2]

        for (x, y, w, h) in faces:
            # Clamp coordinates to bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = rgb_arr[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)

            results.append({
                "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "confidence": 0.90,  # Haar cascade does not provide continuous probabilities
                "face_crop": crop_pil,
                "detector": "haar"
            })

        return results


class MTCNNDetector:
    """Face detector using Multi-task Cascaded Convolutional Networks (MTCNN)."""

    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            select_largest=False,
            device=self.device,
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            thresholds=config.MTCNN_THRESHOLDS,
            post_process=False
        )

    def detect(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detects faces in RGB numpy array or PIL image.
        Returns list of dicts: [{'box': (x, y, w, h), 'confidence': float, 'face_crop': PIL.Image, 'detector': 'mtcnn'}]
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
            rgb_arr = image
        else:
            pil_img = image.convert("RGB")
            rgb_arr = np.array(pil_img)

        boxes, probs = self.mtcnn.detect(pil_img)

        results = []
        if boxes is None or len(boxes) == 0:
            return results

        h_img, w_img = rgb_arr.shape[:2]

        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.5:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Ensure coordinates are within image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            w = x2 - x1
            h = y2 - y1

            if w <= 10 or h <= 10:
                continue

            crop = rgb_arr[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)

            results.append({
                "box": (x1, y1, w, h),
                "confidence": float(prob),
                "face_crop": crop_pil,
                "detector": "mtcnn"
            })

        return results



def get_face_detector(detector_type: str = config.DEFAULT_DETECTOR):
    """
    Factory function returning the selected face detector.
    Options: 'mtcnn' (default, higher precision) or 'haar' (faster, classic).
    """
    detector_type = detector_type.lower()
    if detector_type == "mtcnn":
        return MTCNNDetector()
    elif detector_type in ["haar", "cascade"]:
        return HaarCascadeDetector()
    else:
        raise ValueError(f"Unknown detector type '{detector_type}'. Choose 'mtcnn' or 'haar'.")


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draws bounding boxes and labels onto an OpenCV RGB/BGR image for visualization.
    """
    annotated = image.copy()
    for idx, det in enumerate(detections):
        x, y, w, h = det["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label_text = det.get("detector", "Face")
        if labels and idx < len(labels):
            label_text = labels[idx]
        elif "confidence" in det:
            label_text = f"{det['detector'].upper()}: {det['confidence']:.2f}"

        # Background rectangle for text
        cv2.rectangle(annotated, (x, max(0, y - 24)), (x + w, y), color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x + 5, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )
    return annotated
