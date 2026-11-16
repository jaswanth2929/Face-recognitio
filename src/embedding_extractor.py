"""
Facial Embedding Extraction Module using Pretrained FaceNet (InceptionResnetV1).
Fulfills Step 5: Extract facial embeddings using pretrained models (FaceNet / Dlib).
"""

from typing import List, Union
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

import config
from src.preprocessor import FacePreprocessor


class FaceNetExtractor:
    """
    Extracts deep 512-dimensional facial feature representations (embeddings)
    using FaceNet pretrained on VGGFace2.
    """

    def __init__(self, pretrained: str = "vggface2", device: str = config.DEVICE):
        self.device = device
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        self.preprocessor = FacePreprocessor(target_size=config.FACE_IMAGE_SIZE)

    def extract_embedding(self, face_image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extracts a single 512-d L2-normalized embedding vector for a given face crop.
        Returns 1D numpy array of shape (512,).
        """
        if isinstance(face_image, np.ndarray):
            pil_img = Image.fromarray(face_image).convert("RGB")
        else:
            pil_img = face_image.convert("RGB")

        # Resize and center crop
        processed_pil = self.preprocessor.resize_and_crop(pil_img)
        # Normalize to tensor [1, 3, 160, 160]
        tensor = self.preprocessor.normalize_for_model(processed_pil).to(self.device)

        with torch.no_grad():
            raw_emb = self.model(tensor)
            # L2 normalize
            norm_emb = torch.nn.functional.normalize(raw_emb, p=2, dim=1)
            return norm_emb.squeeze(0).cpu().numpy()

    def extract_embeddings_batch(self, face_images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Extracts embeddings for a batch of face images.
        Returns 2D numpy array of shape (N, 512).
        """
        if not face_images:
            return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

        tensors = []
        for img in face_images:
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            else:
                pil_img = img.convert("RGB")
            processed_pil = self.preprocessor.resize_and_crop(pil_img)
            tensors.append(self.preprocessor.to_tensor_transform(processed_pil))

        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            raw_emb = self.model(batch_tensor)
            norm_emb = torch.nn.functional.normalize(raw_emb, p=2, dim=1)
            return norm_emb.cpu().numpy()

    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes cosine similarity between two L2-normalized embeddings.
        Returns value in [-1, 1], where 1 represents identity match.
        """
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        return float(np.dot(e1, e2))
