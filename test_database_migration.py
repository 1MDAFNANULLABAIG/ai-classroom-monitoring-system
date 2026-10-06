"""
Regression Test Suite for Database Schema & Migration System
Verifies:
1. Legacy students table without 'usn' gets safely migrated.
2. Existing student records preserve data and get 'usn' backfilled from roll_no.
3. Student registration via API succeeds without "no column named usn" error.
4. Classes, activeness_alerts, and classroom intelligence tables are migrated properly.
5. Migration is idempotent and safe to run repeatedly.
"""

import os
import sys
import tempfile
import unittest
import sqlite3
import json

import app
import config
from werkzeug.security import generate_password_hash


class TestDatabaseMigration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.test_dir, "legacy_test.db")
        self.old_db_path = app.DB_PATH
        self.old_config_db_path = config.DB_PATH
        app.DB_PATH = self.test_db
        config.DB_PATH = self.test_db
        app._db_initialized = False
        app.app.config["TESTING"] = True
        app.app.config["SECRET_KEY"] = "migration-test-secret"
        self.client = app.app.test_client()

    def tearDown(self):
        app.DB_PATH = self.old_db_path
        config.DB_PATH = self.old_config_db_path
        app._db_initialized = False
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except OSError:
                pass
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except OSError:
                pass

    def test_fresh_database_has_all_modern_columns_and_tables(self):
        """A fresh database initialization must contain all modern columns."""
        app.init_db()

        conn = sqlite3.connect(self.test_db)
        conn.row_factory = sqlite3.Row

        # 1. Students columns
        student_cols = {r["name"] for r in conn.execute("PRAGMA table_info(students)").fetchall()}
        required_student_cols = {
            "id", "roll_no", "name", "usn", "class_id", "email", "phone",
            "department", "current_semester", "section", "admission_year",
            "status", "status_set_at", "face_deleted_at"
        }
        self.assertTrue(required_student_cols.issubset(student_cols),
                        f"Missing student cols: {required_student_cols - student_cols}")

        # 2. Classes columns
        class_cols = {r["name"] for r in conn.execute("PRAGMA table_info(classes)").fetchall()}
        required_class_cols = {"id", "name", "start_time", "end_time", "late_after_minutes", "department", "semester"}
        self.assertTrue(required_class_cols.issubset(class_cols),
                        f"Missing class cols: {required_class_cols - class_cols}")

        # 3. Activeness alerts columns
        alert_cols = {r["name"] for r in conn.execute("PRAGMA table_info(activeness_alerts)").fetchall()}
        required_alert_cols = {
            "id", "session_id", "student_id", "student_name", "usn", "alert_type",
            "severity", "detail", "photo_path", "video_path", "review_status",
            "reviewed_by", "reviewed_at", "review_notes", "created_at"
        }
        self.assertTrue(required_alert_cols.issubset(alert_cols),
                        f"Missing alert cols: {required_alert_cols - alert_cols}")

        # 4. Intelligence tables
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        required_tables = {"student_activity", "classroom_alerts", "classroom_metrics", "behavior_predictions"}
        self.assertTrue(required_tables.issubset(tables),
                        f"Missing tables: {required_tables - tables}")

        conn.close()

    def test_legacy_database_migration_and_student_registration(self):
        """
        Regression Test:
        Simulate an older legacy database that was created WITHOUT the 'usn' column.
        Verify that migration adds 'usn', preserves data, backfills usn, and allows student registration.
        """
        # Step 1: Create a legacy database manually with the OLD schema
        conn = sqlite3.connect(self.test_db)
        conn.execute("""
            CREATE TABLE classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT NOT NULL,
                name TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                email TEXT,
                phone TEXT,
                FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
                UNIQUE(roll_no, class_id)
            )
        """)
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'admin'
            )
        """)
        # Insert admin user
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin")
        )
        # Insert a class
        conn.execute("INSERT INTO classes (name) VALUES ('CSE-A')")
        # Insert a legacy student WITHOUT usn
        conn.execute("INSERT INTO students (roll_no, name, class_id, email) VALUES ('1HK23IS001', 'Alice Legacy', 1, 'alice@test.edu')")
        conn.commit()

        # Confirm that 'usn' does NOT exist in the legacy students table
        cols_before = {r[1] for r in conn.execute("PRAGMA table_info(students)").fetchall()}
        self.assertNotIn("usn", cols_before)
        conn.close()

        # Step 2: Run automatic database initialization/migration
        app.ensure_db_initialized()

        # Step 3: Verify 'usn' was added and legacy student was backfilled
        conn = sqlite3.connect(self.test_db)
        conn.row_factory = sqlite3.Row
        cols_after = {r["name"] for r in conn.execute("PRAGMA table_info(students)").fetchall()}
        self.assertIn("usn", cols_after)
        self.assertIn("department", cols_after)
        self.assertIn("current_semester", cols_after)
        self.assertIn("section", cols_after)
        self.assertIn("admission_year", cols_after)

        legacy_st = conn.execute("SELECT * FROM students WHERE roll_no = '1HK23IS001'").fetchone()
        self.assertIsNotNone(legacy_st)
        self.assertEqual(legacy_st["name"], "Alice Legacy")
        self.assertEqual(legacy_st["usn"], "1HK23IS001", "Existing student USN should be backfilled from roll_no")
        conn.close()

        # Step 4: Login as admin to get auth token
        login_res = self.client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        self.assertEqual(login_res.status_code, 200)
        token = login_res.get_json()["token"]
        auth_headers = {"Authorization": f"Bearer {token}"}

        # Step 5: Register a NEW student through the production API endpoint /api/students
        # This was the exact operation failing on production with:
        # "Error registering student: table students has no column named usn"
        reg_payload = {
            "name": "Bob Modern",
            "roll_no": "1HK23IS002",
            "usn": "1HK23IS002",
            "class_id": 1,
            "department": "ISE",
            "semester": 5,
            "section": "A",
            "admission_year": "2023",
            "email": "bob@test.edu",
            "phone": "9876543210"
        }
        reg_res = self.client.post("/api/students", json=reg_payload, headers=auth_headers)
        self.assertEqual(reg_res.status_code, 200, f"Registration response: {reg_res.get_data(as_text=True)}")
        reg_json = reg_res.get_json()
        self.assertTrue(reg_json.get("success"))
        new_student_id = reg_json.get("id")
        self.assertIsNotNone(new_student_id)

        # Step 6: Verify new student in database
        conn = sqlite3.connect(self.test_db)
        conn.row_factory = sqlite3.Row
        new_st = conn.execute("SELECT * FROM students WHERE id = ?", (new_student_id,)).fetchone()
        self.assertIsNotNone(new_st)
        self.assertEqual(new_st["name"], "Bob Modern")
        self.assertEqual(new_st["roll_no"], "1HK23IS002")
        self.assertEqual(new_st["usn"], "1HK23IS002")
        self.assertEqual(new_st["department"], "ISE")
        self.assertEqual(new_st["current_semester"], 5)
        self.assertEqual(new_st["section"], "A")
        self.assertEqual(new_st["status"], "active")
        conn.close()

    def test_migration_is_idempotent(self):
        """Calling init_db multiple times must never corrupt data or fail."""
        app.init_db()
        app.init_db()
        app.init_db()

        conn = sqlite3.connect(self.test_db)
        conn.row_factory = sqlite3.Row
        users_count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        classes_count = conn.execute("SELECT COUNT(*) AS c FROM classes").fetchone()["c"]
        self.assertEqual(users_count, 1)
        self.assertEqual(classes_count, 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
