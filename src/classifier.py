"""
Face Classification Module using SVM and KNN classifiers.
Fulfills Step 6: Train classifier (SVM / KNN) on embeddings.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config


class FaceClassifier:
    """Trains, saves, loads, and infers face identity using SVM or KNN on FaceNet embeddings."""

    def __init__(self, model_type: str = config.DEFAULT_CLASSIFIER):
        self.model_type = model_type.lower()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _init_model(self, model_type: str):
        if model_type == "svm":
            return SVC(
                kernel=config.SVM_KERNEL,
                C=config.SVM_C,
                probability=True,
                random_state=42
            )
        elif model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=config.KNN_N_NEIGHBORS,
                weights="distance",
                metric="cosine"
            )
        else:
            raise ValueError(f"Unsupported classifier type '{model_type}'. Choose 'svm' or 'knn'.")

    def train(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        test_size: float = 0.25,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trains the classifier on embedding vectors and labels.
        Returns evaluation metrics dictionary.
        """
        if model_type:
            self.model_type = model_type.lower()

        if len(labels) < 2:
            raise ValueError("Training requires at least 2 samples.")

        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            raise ValueError("Training requires at least 2 distinct student classes.")

        # Encode categorical labels to integers
        y_encoded = self.label_encoder.fit_transform(labels)
        X = np.array(embeddings)

        # Train/Test Split
        # If any class has only 1 sample, fallback to full train evaluation
        class_counts = {lbl: labels.count(lbl) for lbl in unique_labels}
        can_stratify = all(count >= 2 for count in class_counts.values())

        if can_stratify and len(labels) >= 6:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded

        # Instantiate and fit model
        self.model = self._init_model(self.model_type)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Predictions on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "model_type": self.model_type,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classes": self.label_encoder.classes_.tolist(),
            "report": report,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # Save model and encoder
        self.save()
        return metrics

    def predict(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Predicts identity and confidence for a single 512-d embedding.
        Returns (predicted_student_id_or_name, confidence_percentage)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Classifier is not trained. Train or load weights first.")

        emb_2d = embedding.reshape(1, -1)
        probabilities = self.model.predict_proba(emb_2d)[0]
        best_idx = np.argmax(probabilities)
        confidence = float(probabilities[best_idx])
        predicted_label = self.label_encoder.inverse_transform([best_idx])[0]

        return predicted_label, confidence

    def save(self, filepath: Optional[Path] = None):
        """Saves model weights and label encoder to disk."""
        if filepath is None:
            filepath = config.MODELS_DIR / f"classifier_{self.model_type}.joblib"
        payload = {
            "model_type": self.model_type,
            "model": self.model,
            "label_encoder": self.label_encoder,
            "is_trained": self.is_trained
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: Optional[Path] = None, model_type: Optional[str] = None):
        """Loads model weights and label encoder from disk."""
        if model_type:
            self.model_type = model_type.lower()
        if filepath is None:
            filepath = config.MODELS_DIR / f"classifier_{self.model_type}.joblib"

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")

        payload = joblib.load(filepath)
        self.model_type = payload["model_type"]
        self.model = payload["model"]
        self.label_encoder = payload["label_encoder"]
        self.is_trained = payload["is_trained"]
        return self


def compare_classifiers(embeddings: np.ndarray, labels: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Trains both SVM and KNN classifiers on the same embeddings and compares their performance.
    """
    results = {}
    for mtype in ["svm", "knn"]:
        clf = FaceClassifier(model_type=mtype)
        metrics = clf.train(embeddings, labels, model_type=mtype)
        results[mtype] = metrics
    return results
