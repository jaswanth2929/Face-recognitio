"""
Image Preprocessing and Data Augmentation Module.
Fulfills Step 3: Preprocess images - resize, normalize, and augment dataset.
"""

import random
from typing import List, Tuple, Union
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms

import config


class FacePreprocessor:
    """Handles image resizing, normalization, and data augmentation for face recognition."""

    def __init__(self, target_size: Tuple[int, int] = config.FACE_IMAGE_SIZE):
        self.target_size = target_size
        # Standard FaceNet tensor normalization: mean=0.5, std=0.5 maps [0, 1] to [-1, 1]
        self.to_tensor_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def resize_and_crop(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """
        Resize image to target size while preserving aspect ratio and center-cropping.
        """
        if isinstance(image, np.ndarray):
            # Check if OpenCV BGR, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size
        target_w, target_h = self.target_size

        # Aspect ratio preserving resize
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        return resized.crop((left, top, right, bottom))

    def normalize_for_model(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Converts PIL image to normalized PyTorch tensor ready for FaceNet inference.
        Returns tensor of shape [1, 3, H, W].
        """
        tensor = self.to_tensor_transform(pil_image)
        return tensor.unsqueeze(0)

    @staticmethod
    def augment_image(pil_image: Image.Image) -> Image.Image:
        """
        Applies a randomized combination of data augmentations:
        - Horizontal flip (50% probability)
        - Rotation between -15° and +15°
        - Brightness variation (0.75x to 1.25x)
        - Contrast variation (0.8x to 1.2x)
        - Color saturation variation (0.8x to 1.2x)
        """
        img = pil_image.copy()

        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Random Rotation
        angle = random.uniform(-15.0, 15.0)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

        # 3. Random Brightness Jitter
        brightness_factor = random.uniform(0.75, 1.25)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

        # 4. Random Contrast Jitter
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # 5. Random Color Saturation
        color_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(color_factor)

        return img

    def generate_augmented_dataset(
        self,
        images: List[Image.Image],
        target_count_per_person: int = 15
    ) -> List[Image.Image]:
        """
        Takes a list of base images for an individual and generates augmented variations
        until the total count reaches `target_count_per_person`.
        """
        augmented = list(images)
        if not images:
            return augmented

        idx = 0
        while len(augmented) < target_count_per_person:
            base_img = images[idx % len(images)]
            aug_img = self.augment_image(base_img)
            augmented.append(aug_img)
            idx += 1

        return augmented
