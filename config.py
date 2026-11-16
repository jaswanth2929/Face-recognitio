"""
Configuration module for Face Recognition Attendance System.
Handles all paths, model hyperparameters, detection settings, and attendance rules.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
STUDENTS_DIR = DATA_DIR / "students"
RAW_DATASET_DIR = DATA_DIR / "raw"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ATTENDANCE_DIR = BASE_DIR / "attendance_records"

# Ensure essential directories exist
for directory in [DATA_DIR, STUDENTS_DIR, RAW_DATASET_DIR, TEST_SAMPLES_DIR, MODELS_DIR, LOGS_DIR, ATTENDANCE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Links (as specified in project requirements)
DATASET_LINKS = {
    "CelebA": "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset",
    "Faces in the Wild (LFW)": "https://www.kaggle.com/datasets/andrewmvd/faces-in-the-wild",
    "LFW Direct Mirror": "http://vis-www.cs.umass.edu/lfw/"
}

# Image Preprocessing & FaceNet Settings
FACE_IMAGE_SIZE = (160, 160)  # Standard input resolution for FaceNet
EMBEDDING_DIM = 512
DEVICE = "cpu"  # Torch device ("cpu" or "mps")

# Face Detection Settings
DEFAULT_DETECTOR = "mtcnn"  # "mtcnn" or "haar"
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)
MTCNN_MIN_FACE_SIZE = 20
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]

# Classifier Settings
DEFAULT_CLASSIFIER = "svm"  # "svm" or "knn"
SVM_KERNEL = "linear"  # "linear" or "rbf"
SVM_C = 1.0
KNN_N_NEIGHBORS = 3
RECOGNITION_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to mark attendance

# Attendance Tracking Settings
ATTENDANCE_DB_PATH = ATTENDANCE_DIR / "attendance.db"
ATTENDANCE_CSV_PATH = ATTENDANCE_DIR / "attendance_log.csv"
SNAPSHOTS_DIR = ATTENDANCE_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cooldown to avoid duplicate entries for the same student on the same day (in minutes, or "daily")
ATTENDANCE_COOLDOWN_MINUTES = 30
