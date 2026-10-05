"""
Comprehensive Test Suite for Teacher-First Attendance State Machine
Tests 1 through 10 as specified in the Phase 2 requirements.
"""

import os
import sys
import tempfile
import sqlite3
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import app as app_module
from app import app, init_db


class TeacherFirstAttendanceTests(unittest.TestCase):
    def setUp(self):
        self.old_db_path = app_module.DB_PATH
        self.test_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.test_dir, "test_attendance.db")
        app_module.DB_PATH = self.test_db
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret"
        
        # Initialize test database
        init_db()
        self.conn = sqlite3.connect(self.test_db)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        self.conn.execute("DELETE FROM period_attendance")
        self.conn.execute("DELETE FROM period_sessions")
        self.conn.execute("DELETE FROM timetable")
        self.conn.execute("DELETE FROM students")
        self.conn.execute("DELETE FROM teachers")
        self.conn.execute("DELETE FROM classes")

        self.conn.execute("INSERT INTO classes (id, name) VALUES (1, 'Class 10-A')")
        
        # Teacher 1 (Alice, assigned to Period 1)
        self.conn.execute("INSERT INTO teachers (id, teacher_code, name) VALUES (1, 'T001', 'Alice')")
        # Teacher 2 (Bob, assigned to Period 2)
        self.conn.execute("INSERT INTO teachers (id, teacher_code, name) VALUES (2, 'T002', 'Bob')")

        # Students (Charlie, David)
        self.conn.execute("INSERT INTO students (id, roll_no, name, class_id) VALUES (1, 'S001', 'Charlie', 1)")
        self.conn.execute("INSERT INTO students (id, roll_no, name, class_id) VALUES (2, 'S002', 'David', 1)")

        # Timetable:
        # Period 1: 09:00 - 09:50, Class, Subject: Math, Teacher: 1 (Alice)
        # Period 2: 10:00 - 10:50, Class, Subject: Science, Teacher: 2 (Bob)
        self.conn.execute("""
            INSERT INTO timetable (id, class_id, period_number, period_type, subject, teacher_id, start_time, end_time)
            VALUES (1, 1, 1, 'class', 'Math', 1, '09:00', '09:50')
        """)
        self.conn.execute("""
            INSERT INTO timetable (id, class_id, period_number, period_type, subject, teacher_id, start_time, end_time)
            VALUES (2, 1, 2, 'class', 'Science', 2, '10:00', '10:50')
        """)
        self.conn.commit()

        # Set mock recognizers so models appear trained
        app_module.teacher_recognizer = MagicMock()
        app_module.recognizer = MagicMock()

        self.client = app.test_client()
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "admin"

    def tearDown(self):
        self.conn.close()
        app_module.teacher_recognizer = None
        app_module.recognizer = None
        app_module.DB_PATH = getattr(self, "old_db_path", app_module.DB_PATH)
        try:
            if os.path.exists(self.test_db):
                os.remove(self.test_db)
        except Exception:
            pass

    def dummy_frame(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # -------------------------------------------------------------
    # TEST 1: Period starts. Teacher absent. Student attempts attendance at +1 min.
    # EXPECTED: REJECTED (403/400).
    # -------------------------------------------------------------
    def test_01_period_starts_teacher_absent_student_marking_rejected(self):
        test_now = datetime(2026, 9, 9, 9, 1, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Check state
            resp = self.client.get("/attendance/smart/state?class_id=1")
            data = resp.get_json()
            self.assertEqual(data["state"], "WAITING_FOR_TEACHER")
            self.assertFalse(data["student_attendance_allowed"])
            self.assertEqual(data["teacher_status"], "none")

            # Try to mark student directly with invalid/premature session
            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 50, 50)]
                app_module.recognizer.predict.return_value = (1, 30.0)

                resp_mark = self.client.post("/attendance/smart/mark_student", json={
                    "session_id": 999,
                    "image": mock_img
                })
                self.assertEqual(resp_mark.status_code, 404)

                resp_live = self.client.post("/attendance/live/recognize", json={
                    "class_id": 1,
                    "att_date": "2026-09-09",
                    "image": mock_img
                })
                self.assertEqual(resp_live.status_code, 403)
                self.assertFalse(resp_live.get_json()["success"])
                self.assertEqual(resp_live.get_json()["state"], "WAITING_FOR_TEACHER")

    # -------------------------------------------------------------
    # TEST 2: Teacher recognized at +5 minutes.
    # EXPECTED: Teacher verified and student attendance enabled.
    # -------------------------------------------------------------
    def test_02_teacher_recognized_at_plus_5_minutes(self):
        test_now = datetime(2026, 9, 9, 9, 5, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 80, 80)]
                app_module.teacher_recognizer.predict.return_value = (1, 40.0)  # Alice

                resp = self.client.post("/attendance/smart/start_session", json={
                    "class_id": 1,
                    "image": mock_img
                })
                data = resp.get_json()
                self.assertTrue(data["success"])
                self.assertEqual(data["state"], "TEACHER_CONFIRMED")
                self.assertEqual(data["teacher_name"], "Alice")
                session_id = data["session_id"]

            state_resp = self.client.get("/attendance/smart/state?class_id=1")
            state_data = state_resp.get_json()
            self.assertEqual(state_data["state"], "TEACHER_CONFIRMED")
            self.assertTrue(state_data["student_attendance_allowed"])
            self.assertEqual(state_data["teacher_status"], "present")

            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det, patch("app.better_teacher_match") as mock_bt:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 50, 50)]
                app_module.recognizer.predict.return_value = (1, 35.0)  # Charlie
                mock_bt.return_value = None

                mark_resp = self.client.post("/attendance/smart/mark_student", json={
                    "session_id": session_id,
                    "image": mock_img
                })
                mark_data = mark_resp.get_json()
                self.assertTrue(mark_data["success"])
                self.assertEqual(mark_data["present_count"], 1)

    # -------------------------------------------------------------
    # TEST 3: Teacher recognized at +29 minutes.
    # EXPECTED: Student attendance enabled (still within 30-minute grace period).
    # -------------------------------------------------------------
    def test_03_teacher_recognized_at_plus_29_minutes(self):
        test_now = datetime(2026, 9, 9, 9, 29, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 80, 80)]
                app_module.teacher_recognizer.predict.return_value = (1, 42.0)

                resp = self.client.post("/attendance/smart/start_session", json={
                    "class_id": 1,
                    "image": mock_img
                })
                data = resp.get_json()
                self.assertTrue(data["success"])
                self.assertEqual(data["state"], "TEACHER_CONFIRMED")
                self.assertEqual(data["teacher_status"], "present")

            state_resp = self.client.get("/attendance/smart/state?class_id=1")
            self.assertTrue(state_resp.get_json()["student_attendance_allowed"])

    # -------------------------------------------------------------
    # TEST 4: Teacher absent at +30 minutes.
    # EXPECTED: Teacher absent policy activates and automatic student attendance rule executes.
    # -------------------------------------------------------------
    def test_04_teacher_absent_at_plus_30_minutes(self):
        test_now = datetime(2026, 9, 9, 9, 30, 0)  # Exactly +30 minutes from 09:00 start
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            state_resp = self.client.get("/attendance/smart/state?class_id=1")
            state_data = state_resp.get_json()
            self.assertEqual(state_data["state"], "TEACHER_ABSENT_AUTO_ATTENDANCE")
            self.assertEqual(state_data["teacher_status"], "absent")
            self.assertTrue(state_data["student_attendance_allowed"])

            session_id = state_data["session"]["id"]
            self.assertIsNotNone(session_id)
            self.assertIsNone(state_data["session"]["teacher_name"])

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det, patch("app.better_teacher_match") as mock_bt:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 50, 50)]
                app_module.recognizer.predict.return_value = (2, 38.0)  # David
                mock_bt.return_value = None

                mark_resp = self.client.post("/attendance/smart/mark_student", json={
                    "session_id": session_id,
                    "image": mock_img
                })
                mark_data = mark_resp.get_json()
                self.assertTrue(mark_data["success"])
                self.assertEqual(mark_data["present_count"], 1)
                self.assertEqual(mark_data["faces"][0]["name"], "David")

    # -------------------------------------------------------------
    # TEST 5: Browser refresh at +15 minutes.
    # EXPECTED: Timer remains based on scheduled period start (15 min remaining).
    # -------------------------------------------------------------
    def test_05_browser_refresh_at_plus_15_minutes(self):
        test_now = datetime(2026, 9, 9, 9, 15, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            resp1 = self.client.get("/attendance/smart/state?class_id=1")
            data1 = resp1.get_json()
            self.assertEqual(data1["state"], "WAITING_FOR_TEACHER")
            self.assertEqual(data1["remaining_seconds"], 15 * 60)

            resp2 = self.client.get("/attendance/smart/state?class_id=1")
            data2 = resp2.get_json()
            self.assertEqual(data2["state"], "WAITING_FOR_TEACHER")
            self.assertEqual(data2["remaining_seconds"], 15 * 60)
            self.assertEqual(data2["grace_deadline"], "09:30")
            self.assertEqual(data2["period_start"], "09:00")

    # -------------------------------------------------------------
    # TEST 6: Student tries to bypass frontend and directly call attendance API before teacher confirmation.
    # EXPECTED: Backend rejects request.
    # -------------------------------------------------------------
    def test_06_student_attempts_direct_api_bypass_before_teacher(self):
        test_now = datetime(2026, 9, 9, 9, 10, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            # Directly call smart_mark_student without valid session
            resp_smart = self.client.post("/attendance/smart/mark_student", json={
                "session_id": 9999,
                "image": mock_img
            })
            self.assertEqual(resp_smart.status_code, 404)

            # Directly call live_recognize without teacher session
            resp_live = self.client.post("/attendance/live/recognize", json={
                "class_id": 1,
                "att_date": "2026-09-09",
                "image": mock_img
            })
            self.assertEqual(resp_live.status_code, 403)
            self.assertEqual(resp_live.get_json()["state"], "WAITING_FOR_TEACHER")

    # -------------------------------------------------------------
    # TEST 7: Teacher arrives at +35 minutes.
    # EXPECTED: No timer reset; late/absent policy is followed (teacher_status='late').
    # -------------------------------------------------------------
    def test_07_teacher_arrives_at_plus_35_minutes(self):
        # Auto-attendance triggered at 09:30
        auto_now = datetime(2026, 9, 9, 9, 30, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = auto_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            self.client.get("/attendance/smart/state?class_id=1")

        # Teacher arrives at 09:35 (+35 min)
        late_now = datetime(2026, 9, 9, 9, 35, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = late_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 80, 80)]
                app_module.teacher_recognizer.predict.return_value = (1, 39.0)

                resp = self.client.post("/attendance/smart/start_session", json={
                    "class_id": 1,
                    "image": mock_img
                })
                data = resp.get_json()
                self.assertTrue(data["success"])
                self.assertEqual(data["state"], "TEACHER_LATE")
                self.assertEqual(data["teacher_status"], "late")

            state_resp = self.client.get("/attendance/smart/state?class_id=1")
            state_data = state_resp.get_json()
            self.assertEqual(state_data["state"], "TEACHER_LATE")
            self.assertEqual(state_data["teacher_status"], "late")
            self.assertTrue(state_data["student_attendance_allowed"])

    # -------------------------------------------------------------
    # TEST 8: Two requests attempt to mark the same student simultaneously.
    # EXPECTED: Only one attendance record.
    # -------------------------------------------------------------
    def test_08_duplicate_marking_prevention(self):
        test_now = datetime(2026, 9, 9, 9, 5, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 80, 80)]
                app_module.teacher_recognizer.predict.return_value = (1, 40.0)
                resp = self.client.post("/attendance/smart/start_session", json={"class_id": 1, "image": mock_img})
                session_id = resp.get_json()["session_id"]

            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det, patch("app.better_teacher_match") as mock_bt:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 50, 50)]
                app_module.recognizer.predict.return_value = (1, 35.0)  # Charlie
                mock_bt.return_value = None

                r1 = self.client.post("/attendance/smart/mark_student", json={"session_id": session_id, "image": mock_img})
                self.assertEqual(r1.get_json()["faces"][0]["status"], "marked")

                r2 = self.client.post("/attendance/smart/mark_student", json={"session_id": session_id, "image": mock_img})
                self.assertEqual(r2.get_json()["faces"][0]["status"], "already_marked")

            count = self.conn.execute("SELECT COUNT(*) FROM period_attendance WHERE session_id = ? AND student_id = 1", (session_id,)).fetchone()[0]
            self.assertEqual(count, 1)

    # -------------------------------------------------------------
    # TEST 9: Wrong teacher attempts to start the period.
    # EXPECTED: Teacher rejected.
    # -------------------------------------------------------------
    def test_09_wrong_teacher_rejected(self):
        test_now = datetime(2026, 9, 9, 9, 5, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 80, 80)]
                app_module.teacher_recognizer.predict.return_value = (2, 40.0)  # Bob (not assigned to Period 1)

                resp = self.client.post("/attendance/smart/start_session", json={
                    "class_id": 1,
                    "image": mock_img
                })
                data = resp.get_json()
                self.assertFalse(data["success"])
                self.assertIn("not the teacher scheduled for this period", data["message"])

    # -------------------------------------------------------------
    # TEST 10: Teacher belongs to another timetable period.
    # EXPECTED: Teacher rejected.
    # -------------------------------------------------------------
    def test_10_teacher_belongs_to_another_period_rejected(self):
        test_now = datetime(2026, 9, 9, 9, 5, 0)
        with patch("app.datetime") as mock_dt:
            mock_dt.now.return_value = test_now
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            mock_img = "data:image/jpeg;base64,ZmFrZQ=="
            with patch("app.decode_base64_image") as mock_dec, patch("app.detect_faces") as mock_det:
                mock_dec.return_value = self.dummy_frame()
                mock_det.return_value = [(10, 10, 80, 80)]
                app_module.teacher_recognizer.predict.return_value = (2, 35.0)

                resp = self.client.post("/attendance/smart/start_session", json={
                    "class_id": 1,
                    "image": mock_img
                })
                data = resp.get_json()
                self.assertFalse(data["success"])
                self.assertIn("Bob is not the teacher scheduled for this period", data["message"])


if __name__ == "__main__":
    unittest.main()
