"""
Attendance Marking and Record Management System.
Fulfills Step 7: Develop attendance marking system integrating recognition and timestamp.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
from PIL import Image

import config


class AttendanceManager:
    """
    Manages student attendance recording, SQLite database storage,
    timestamping, deduplication/cooldown, and CSV synchronization.
    """

    def __init__(
        self,
        db_path: Path = config.ATTENDANCE_DB_PATH,
        csv_path: Path = config.ATTENDANCE_CSV_PATH,
        cooldown_minutes: int = config.ATTENDANCE_COOLDOWN_MINUTES
    ):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.cooldown_minutes = cooldown_minutes
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initializes the SQLite attendance database and table schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    department TEXT DEFAULT 'Computer Science',
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Present',
                    detector_used TEXT DEFAULT 'mtcnn',
                    snapshot_path TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_student_date 
                ON attendance (student_id, date)
            """)
            conn.commit()

    def mark_attendance(
        self,
        student_id: str,
        student_name: str,
        confidence: float,
        detector_used: str = "mtcnn",
        department: str = "Computer Science",
        snapshot_image: Optional[Image.Image] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Marks attendance for a recognized student with timestamp verification and deduplication.
        Returns: (success: bool, message: str, record_details: dict)
        """
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Check duplicate attendance within cooldown period
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp FROM attendance 
                WHERE student_id = ? AND date = ? 
                ORDER BY id DESC LIMIT 1
            """, (student_id, current_date))
            last_record = cursor.fetchone()

            if last_record:
                last_time_str = last_record["timestamp"]
                last_dt = datetime.strptime(f"{current_date} {last_time_str}", "%Y-%m-%d %H:%M:%S")
                time_diff = (now - last_dt).total_seconds() / 60.0

                if time_diff < self.cooldown_minutes:
                    return (
                        False,
                        f"Duplicate skipped: {student_name} ({student_id}) already marked today at {last_time_str} (Cooldown: {self.cooldown_minutes}m)",
                        {"student_id": student_id, "student_name": student_name, "last_marked": last_time_str}
                    )

            # Save snapshot if provided
            snapshot_filename = ""
            if snapshot_image:
                config.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                snapshot_filename = f"{student_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                snapshot_full_path = config.SNAPSHOTS_DIR / snapshot_filename
                snapshot_image.save(snapshot_full_path, "JPEG")

            # Insert attendance record
            cursor.execute("""
                INSERT INTO attendance (
                    student_id, student_name, department, date, 
                    timestamp, confidence, status, detector_used, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_id,
                student_name,
                department,
                current_date,
                current_time,
                round(confidence, 4),
                "Present",
                detector_used,
                snapshot_filename
            ))
            conn.commit()

        # Update CSV log
        self._sync_csv()

        record = {
            "student_id": student_id,
            "student_name": student_name,
            "date": current_date,
            "timestamp": current_time,
            "confidence": f"{confidence * 100:.1f}%",
            "status": "Present"
        }
        return (
            True,
            f"Attendance recorded for {student_name} ({student_id}) at {current_time} [Conf: {confidence * 100:.1f}%]",
            record
        )

    def _sync_csv(self):
        """Exports the entire attendance database to CSV."""
        df = self.get_records_dataframe()
        df.to_csv(self.csv_path, index=False)

    def get_records_dataframe(
        self,
        filter_date: Optional[str] = None,
        student_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieves attendance records as a pandas DataFrame with optional filters."""
        query = "SELECT * FROM attendance WHERE 1=1"
        params: List[Any] = []

        if filter_date:
            query += " AND date = ?"
            params.append(filter_date)

        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)

        query += " ORDER BY id DESC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates attendance summary metrics (total records, unique students, daily counts)."""
        df = self.get_records_dataframe()
        if df.empty:
            return {
                "total_records": 0,
                "unique_students_attended": 0,
                "dates_active": 0,
                "average_confidence": 0.0,
                "daily_counts": {},
                "student_counts": {}
            }

        return {
            "total_records": len(df),
            "unique_students_attended": df["student_id"].nunique(),
            "dates_active": df["date"].nunique(),
            "average_confidence": float(df["confidence"].mean()),
            "daily_counts": df["date"].value_counts().to_dict(),
            "student_counts": df["student_name"].value_counts().to_dict()
        }

    def clear_all_records(self):
        """Resets attendance records (useful for testing)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance")
            conn.commit()
        if self.csv_path.exists():
            self.csv_path.unlink()
