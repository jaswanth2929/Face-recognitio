"""
Dataset Loader and Student Management Module.
Fulfills Step 2: Collect face images of students for training, and integrates Kaggle / Open Datasets (LFW & CelebA).
"""

import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

import config
from src.preprocessor import FacePreprocessor


class DatasetManager:
    """Manages student face datasets, enrollments, and Kaggle/LFW/CelebA integrations."""

    def __init__(self, students_dir: Path = config.STUDENTS_DIR):
        self.students_dir = Path(students_dir)
        self.students_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = FacePreprocessor()

    @staticmethod
    def get_dataset_metadata() -> Dict[str, Any]:
        """Returns details and active links for datasets specified in project requirements."""
        return {
            "name": "Face Images Dataset",
            "sources": [
                {
                    "title": "Faces in the Wild (LFW)",
                    "url": config.DATASET_LINKS["Faces in the Wild (LFW)"],
                    "direct_mirror": config.DATASET_LINKS["LFW Direct Mirror"],
                    "description": "13,233 face photographs of 5,749 people collected from the web, labeled with name identities. Ideal for unconstrained face verification & identification benchmarks."
                },
                {
                    "title": "CelebFaces Attributes (CelebA) Dataset",
                    "url": config.DATASET_LINKS["CelebA"],
                    "description": "Large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations and landmark locations."
                }
            ]
        }

    def enroll_student(
        self,
        student_id: str,
        student_name: str,
        images: List[Image.Image],
        department: str = "Computer Science",
        augment_count: int = 15
    ) -> Path:
        """
        Enrolls a new student by creating their folder and saving base + augmented face photos.
        """
        clean_name = "".join(c for c in student_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        folder_name = f"{student_id}_{clean_name}"
        student_dir = self.students_dir / folder_name
        student_dir.mkdir(parents=True, exist_ok=True)

        # Generate augmentations if needed to ensure robust sample representation
        dataset_images = self.preprocessor.generate_augmented_dataset(
            images, target_count_per_person=augment_count
        )

        for idx, img in enumerate(dataset_images):
            filepath = student_dir / f"face_{idx + 1:03d}.jpg"
            img.convert("RGB").save(filepath, "JPEG")

        # Save metadata file
        meta_path = student_dir / "info.txt"
        meta_path.write_text(f"ID: {student_id}\nName: {student_name}\nDepartment: {department}\nSamples: {len(dataset_images)}\n")

        return student_dir

    def list_enrolled_students(self) -> List[Dict[str, Any]]:
        """Lists all registered students with their sample counts and folder paths."""
        students = []
        for path in sorted(self.students_dir.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                parts = path.name.split("_", 1)
                student_id = parts[0]
                student_name = parts[1].replace("_", " ") if len(parts) > 1 else student_id
                image_files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
                students.append({
                    "student_id": student_id,
                    "student_name": student_name,
                    "folder": str(path),
                    "image_count": len(image_files),
                    "first_image": str(image_files[0]) if image_files else None
                })
        return students

    def load_all_training_data(self) -> Tuple[List[Image.Image], List[str]]:
        """
        Scans all student folders and returns a list of PIL Images and their corresponding student labels (ID_Name).
        """
        images = []
        labels = []

        students = self.list_enrolled_students()
        for s in students:
            folder = Path(s["folder"])
            label = f"{s['student_id']}_{s['student_name']}"
            for img_path in sorted(folder.glob("*.*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Skipping corrupt image {img_path}: {e}")

        return images, labels

    def load_sample_students_from_lfw(
        self,
        num_students: int = 5,
        samples_per_student: int = 15
    ) -> List[str]:
        """
        Extracts sample identities from the LFW (Faces in the Wild) dataset
        and creates initial student profiles.
        """
        lfw_dir = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"
        if not lfw_dir.exists():
            # If not yet present, trigger download via scikit-learn
            from sklearn.datasets import fetch_lfw_people
            fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)

        if not lfw_dir.exists():
            raise FileNotFoundError(f"LFW dataset directory not found at {lfw_dir}")

        # Find people with at least `samples_per_student` images
        eligible_people = []
        for person_dir in sorted(lfw_dir.iterdir()):
            if person_dir.is_dir():
                imgs = list(person_dir.glob("*.jpg"))
                if len(imgs) >= 10:
                    eligible_people.append((person_dir.name, imgs))

        if not eligible_people:
            raise RuntimeError("No people found in LFW with sufficient images.")

        enrolled_ids = []
        # Predefined student aliases for clarity in demonstration
        demo_student_aliases = [
            ("STU202601", "Alex Morgan", "Computer Science"),
            ("STU202602", "Elena Vance", "Artificial Intelligence"),
            ("STU202603", "Marcus Reed", "Software Engineering"),
            ("STU202604", "Sophia Chen", "Data Science"),
            ("STU202605", "David Patel", "Information Systems"),
            ("STU202606", "Rachel Green", "Computer Engineering"),
            ("STU202607", "James Wilson", "Robotics"),
            ("STU202608", "Aria Stark", "Cybersecurity")
        ]

        count = min(num_students, len(eligible_people), len(demo_student_aliases))
        for i in range(count):
            stu_id, stu_name, dept = demo_student_aliases[i]
            person_name, img_paths = eligible_people[i]

            raw_imgs = []
            for p in img_paths[:samples_per_student]:
                raw_imgs.append(Image.open(p).convert("RGB"))

            self.enroll_student(
                student_id=stu_id,
                student_name=stu_name,
                images=raw_imgs,
                department=dept,
                augment_count=samples_per_student
            )
            enrolled_ids.append(f"{stu_id}_{stu_name}")

        return enrolled_ids

    def clear_all_students(self):
        """Clears all enrolled students."""
        for item in self.students_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                shutil.rmtree(item)
