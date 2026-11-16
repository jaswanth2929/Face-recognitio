"""
Command-Line Interface (CLI) for Face Recognition Attendance System.
Enables initialization, dataset loading, model training, evaluation, and attendance marking from terminal.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

import config
from src.dataset_loader import DatasetManager
from src.face_detector import get_face_detector
from src.embedding_extractor import FaceNetExtractor
from src.classifier import FaceClassifier, compare_classifiers
from src.attendance_manager import AttendanceManager
from src.robustness_tester import RobustnessTester


def init_dataset(num_students: int = 5):
    print("=" * 60)
    print("INITIALIZING DATASET")
    print("=" * 60)
    dm = DatasetManager()
    meta = dm.get_dataset_metadata()
    print(f"Dataset: {meta['name']}")
    for s in meta["sources"]:
        print(f" - {s['title']}: {s['url']}")
    print("-" * 60)
    print(f"Importing {num_students} students from LFW (Faces in the Wild)...")
    enrolled = dm.load_sample_students_from_lfw(num_students=num_students)
    print(f"Successfully enrolled {len(enrolled)} students:")
    for stu in enrolled:
        print(f"  [+] {stu}")


def train_models():
    print("=" * 60)
    print("EXTRACTING EMBEDDINGS & TRAINING CLASSIFIERS")
    print("=" * 60)
    dm = DatasetManager()
    images, labels = dm.load_all_training_data()
    if not images:
        print("Error: No training images found. Run --init-dataset first.")
        sys.exit(1)

    print(f"Loaded {len(images)} images across {len(set(labels))} students.")
    print("Extracting 512-dimensional FaceNet embeddings...")
    extractor = FaceNetExtractor()
    embeddings = extractor.extract_embeddings_batch(images)
    print(f"Extracted embeddings matrix shape: {embeddings.shape}")

    print("-" * 60)
    print("Comparing SVM and KNN Classifiers...")
    results = compare_classifiers(embeddings, labels)

    for mtype, res in results.items():
        print(f"\n[{mtype.upper()} Classifier Performance]")
        print(f"  Accuracy:  {res['accuracy'] * 100:.2f}%")
        print(f"  Precision: {res['precision'] * 100:.2f}%")
        print(f"  Recall:    {res['recall'] * 100:.2f}%")
        print(f"  F1 Score:  {res['f1_score'] * 100:.2f}%")
        print(f"  Classes:   {res['classes']}")

    print("\nModels successfully saved to 'models/' directory.")


def evaluate_robustness():
    print("=" * 60)
    print("RUNNING ROBUSTNESS BENCHMARK (Lighting, Angles, Blur)")
    print("=" * 60)
    clf = FaceClassifier(model_type="svm")
    clf.load()
    dm = DatasetManager()
    images, labels = dm.load_all_training_data()

    # Sample a subset for quick benchmarking
    test_imgs = images[:20]
    test_lbls = labels[:20]

    tester = RobustnessTester(classifier=clf, detector_type="mtcnn")
    report = tester.generate_robustness_report(test_imgs, test_lbls)

    print("\n[Lighting Benchmark]")
    for k, v in report["lighting"].items():
        print(f"  {k}: Accuracy = {v['accuracy'] * 100:.1f}%, Avg Conf = {v['average_confidence'] * 100:.1f}%")

    print("\n[Angle Benchmark]")
    for k, v in report["angle"].items():
        print(f"  {k}: Accuracy = {v['accuracy'] * 100:.1f}%, Avg Conf = {v['average_confidence'] * 100:.1f}%")

    print("\n[Blur Benchmark]")
    for k, v in report["blur"].items():
        print(f"  {k}: Accuracy = {v['accuracy'] * 100:.1f}%, Avg Conf = {v['average_confidence'] * 100:.1f}%")

    print(f"\nMulti-panel plot saved to: {report['plot_saved_to']}")


def mark_attendance_from_image(image_path: str, detector_type: str = "mtcnn"):
    print(f"Marking attendance using image: {image_path}")
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    image = Image.open(path).convert("RGB")
    detector = get_face_detector(detector_type)
    detections = detector.detect(image)

    if not detections:
        print("No faces detected in the image.")
        return

    print(f"Detected {len(detections)} face(s).")
    extractor = FaceNetExtractor()
    clf = FaceClassifier()
    clf.load()
    att_mgr = AttendanceManager()

    for idx, det in enumerate(detections):
        crop = det["face_crop"]
        emb = extractor.extract_embedding(crop)
        pred_label, conf = clf.predict(emb)
        print(f"Face #{idx + 1}: Predicted '{pred_label}' with {conf * 100:.2f}% confidence.")

        if conf >= config.RECOGNITION_CONFIDENCE_THRESHOLD:
            parts = pred_label.split("_", 1)
            stu_id = parts[0]
            stu_name = parts[1] if len(parts) > 1 else stu_id
            success, msg, _ = att_mgr.mark_attendance(
                student_id=stu_id,
                student_name=stu_name,
                confidence=conf,
                detector_used=detector_type,
                snapshot_image=crop
            )
            print(f"  -> {msg}")
        else:
            print(f"  -> Confidence {conf * 100:.1f}% below threshold {config.RECOGNITION_CONFIDENCE_THRESHOLD * 100}%. Marking skipped.")


def list_attendance():
    att_mgr = AttendanceManager()
    df = att_mgr.get_records_dataframe()
    if df.empty:
        print("No attendance records found.")
        return
    print("=" * 80)
    print("ATTENDANCE RECORDS")
    print("=" * 80)
    print(df.to_string(index=False))
    stats = att_mgr.get_summary_statistics()
    print("-" * 80)
    print(f"Total Entries: {stats['total_records']} | Unique Students: {stats['unique_students_attended']} | Avg Conf: {stats['average_confidence'] * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System CLI")
    parser.add_argument("--init-dataset", action="store_true", help="Initialize demo students from LFW dataset")
    parser.add_argument("--num-students", type=int, default=5, help="Number of students to enroll from LFW")
    parser.add_argument("--train", action="store_true", help="Extract FaceNet embeddings and train SVM/KNN classifiers")
    parser.add_argument("--evaluate-robustness", action="store_true", help="Run robustness tests across lighting, angles, and blur")
    parser.add_argument("--mark-attendance", type=str, metavar="IMAGE_PATH", help="Mark attendance from a photo")
    parser.add_argument("--detector", type=str, default="mtcnn", choices=["mtcnn", "haar"], help="Face detector to use")
    parser.add_argument("--list-attendance", action="store_true", help="Display attendance records table")
    parser.add_argument("--list-students", action="store_true", help="List enrolled students")

    args = parser.parse_args()

    if args.init_dataset:
        init_dataset(args.num_students)
    elif args.train:
        train_models()
    elif args.evaluate_robustness:
        evaluate_robustness()
    elif args.mark_attendance:
        mark_attendance_from_image(args.mark_attendance, args.detector)
    elif args.list_attendance:
        list_attendance()
    elif args.list_students:
        dm = DatasetManager()
        students = dm.list_enrolled_students()
        print(f"Total enrolled students: {len(students)}")
        for s in students:
            print(f"  [{s['student_id']}] {s['student_name']} - {s['image_count']} images")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
