"""
Robustness and Stress Testing Module.
Fulfills Step 9: Test system under different lighting and angles.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import matplotlib.pyplot as plt

import config
from src.face_detector import get_face_detector
from src.embedding_extractor import FaceNetExtractor
from src.classifier import FaceClassifier


class RobustnessTester:
    """
    Evaluates face detection, FaceNet embedding stability, and classifier accuracy
    under synthetic environmental perturbations (illumination, tilt angles, and blur).
    """

    def __init__(
        self,
        classifier: FaceClassifier,
        detector_type: str = "mtcnn",
        device: str = config.DEVICE
    ):
        self.classifier = classifier
        self.detector = get_face_detector(detector_type)
        self.embedding_extractor = FaceNetExtractor(device=device)

    # 1. Illumination Perturbations
    @staticmethod
    def adjust_lighting(image: Image.Image, factor: float) -> Image.Image:
        """Adjusts brightness factor (e.g., 0.3 for low light, 1.8 for harsh glare)."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    # 2. Angle / Pose Perturbations
    @staticmethod
    def rotate_image(image: Image.Image, angle: float) -> Image.Image:
        """Rotates face image by specified degree."""
        return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    # 3. Blur Perturbations
    @staticmethod
    def apply_blur(image: Image.Image, radius: float) -> Image.Image:
        """Applies Gaussian blur with given radius."""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def evaluate_sample(
        self,
        image: Image.Image,
        true_label: str
    ) -> Tuple[bool, float, str]:
        """
        Runs detection -> FaceNet embedding -> Classifier on a single test image.
        Returns: (is_correct: bool, confidence: float, predicted_label: str)
        """
        detections = self.detector.detect(image)
        if not detections:
            # Fallback to center-crop if detector fails on extreme angles
            crop = image
        else:
            crop = detections[0]["face_crop"]

        emb = self.embedding_extractor.extract_embedding(crop)
        pred_label, conf = self.classifier.predict(emb)
        is_correct = (pred_label == true_label)
        return is_correct, conf, pred_label

    def run_lighting_benchmark(
        self,
        test_images: List[Image.Image],
        test_labels: List[str],
        lighting_factors: List[float] = [0.35, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    ) -> Dict[str, Any]:
        """Tests system accuracy and confidence across various lighting multipliers."""
        results = {}
        for factor in lighting_factors:
            correct = 0
            confidences = []
            for img, label in zip(test_images, test_labels):
                perturbed = self.adjust_lighting(img, factor)
                is_correct, conf, _ = self.evaluate_sample(perturbed, label)
                if is_correct:
                    correct += 1
                confidences.append(conf)

            acc = (correct / len(test_images)) if test_images else 0.0
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            results[f"{factor:.2f}x"] = {
                "factor": factor,
                "accuracy": acc,
                "average_confidence": avg_conf,
                "total_tested": len(test_images)
            }
        return results

    def run_angle_benchmark(
        self,
        test_images: List[Image.Image],
        test_labels: List[str],
        angles: List[float] = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    ) -> Dict[str, Any]:
        """Tests system accuracy and confidence across various head tilt angles."""
        results = {}
        for angle in angles:
            correct = 0
            confidences = []
            for img, label in zip(test_images, test_labels):
                perturbed = self.rotate_image(img, angle)
                is_correct, conf, _ = self.evaluate_sample(perturbed, label)
                if is_correct:
                    correct += 1
                confidences.append(conf)

            acc = (correct / len(test_images)) if test_images else 0.0
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            results[f"{int(angle)}°"] = {
                "angle": angle,
                "accuracy": acc,
                "average_confidence": avg_conf,
                "total_tested": len(test_images)
            }
        return results

    def run_blur_benchmark(
        self,
        test_images: List[Image.Image],
        test_labels: List[str],
        radii: List[float] = [0.0, 1.0, 2.0, 3.5, 5.0]
    ) -> Dict[str, Any]:
        """Tests system accuracy and confidence across increasing levels of motion blur."""
        results = {}
        for r in radii:
            correct = 0
            confidences = []
            for img, label in zip(test_images, test_labels):
                perturbed = self.apply_blur(img, r) if r > 0 else img
                is_correct, conf, _ = self.evaluate_sample(perturbed, label)
                if is_correct:
                    correct += 1
                confidences.append(conf)

            acc = (correct / len(test_images)) if test_images else 0.0
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            results[f"Radius {r}"] = {
                "radius": r,
                "accuracy": acc,
                "average_confidence": avg_conf,
                "total_tested": len(test_images)
            }
        return results

    def generate_robustness_report(
        self,
        test_images: List[Image.Image],
        test_labels: List[str],
        save_plot_path: Path = config.LOGS_DIR / "robustness_benchmark.png"
    ) -> Dict[str, Any]:
        """
        Runs comprehensive test suite across lighting, angles, and blur,
        saves a multi-panel plot, and returns aggregate metrics.
        """
        lighting_results = self.run_lighting_benchmark(test_images, test_labels)
        angle_results = self.run_angle_benchmark(test_images, test_labels)
        blur_results = self.run_blur_benchmark(test_images, test_labels)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Face Recognition System Robustness Evaluation", fontsize=15, weight="bold")

        # 1. Lighting Plot
        factors = [v["factor"] for v in lighting_results.values()]
        l_acc = [v["accuracy"] * 100 for v in lighting_results.values()]
        axes[0].plot(factors, l_acc, marker="o", color="#2563EB", linewidth=2.5, markersize=7)
        axes[0].set_title("Accuracy vs. Lighting Factor", fontsize=12)
        axes[0].set_xlabel("Lighting Multiplier (1.0 = Normal)")
        axes[0].set_ylabel("Recognition Accuracy (%)")
        axes[0].set_ylim(-5, 105)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # 2. Angle Plot
        deg = [v["angle"] for v in angle_results.values()]
        a_acc = [v["accuracy"] * 100 for v in angle_results.values()]
        axes[1].plot(deg, a_acc, marker="s", color="#059669", linewidth=2.5, markersize=7)
        axes[1].set_title("Accuracy vs. Head Tilt Angle", fontsize=12)
        axes[1].set_xlabel("Tilt Angle (Degrees)")
        axes[1].set_ylabel("Recognition Accuracy (%)")
        axes[1].set_ylim(-5, 105)
        axes[1].grid(True, linestyle="--", alpha=0.6)

        # 3. Blur Plot
        r_vals = [v["radius"] for v in blur_results.values()]
        b_acc = [v["accuracy"] * 100 for v in blur_results.values()]
        axes[2].plot(r_vals, b_acc, marker="^", color="#DC2626", linewidth=2.5, markersize=7)
        axes[2].set_title("Accuracy vs. Gaussian Blur Radius", fontsize=12)
        axes[2].set_xlabel("Blur Radius (px)")
        axes[2].set_ylabel("Recognition Accuracy (%)")
        axes[2].set_ylim(-5, 105)
        axes[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_plot_path, dpi=200)
        plt.close(fig)

        return {
            "lighting": lighting_results,
            "angle": angle_results,
            "blur": blur_results,
            "plot_saved_to": str(save_plot_path)
        }
