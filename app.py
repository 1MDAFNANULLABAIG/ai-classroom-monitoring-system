"""
Student Attendance Management System
Author: Generated for user
Tech: Flask + SQLite

Run with:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Default admin login:
    Username: admin
    Password: admin123
"""

import os
import time
import glob
import shutil
import json
import random
import base64
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, Response, jsonify,
    g, has_request_context, send_from_directory, abort
)
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
import cv2

import exam_ai
import activeness_ai
import classroom_ai
import behavior_ai
import data_expiry
import config
import face_quality
import face_engine
import tracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")

# Face recognition storage
FACE_DIR = os.path.join(BASE_DIR, "face_data")
DATASET_DIR = os.path.join(FACE_DIR, "dataset")
TRAINER_FILE = os.path.join(FACE_DIR, "trainer.yml")

TEACHER_DATASET_DIR = os.path.join(FACE_DIR, "dataset_teachers")
TEACHER_TRAINER_FILE = os.path.join(FACE_DIR, "trainer_teachers.yml")

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# How many good face samples we ask for per student/teacher during capture (Item 1, 15)
SAMPLES_PER_STUDENT = config.SAMPLES_PER_STUDENT
SAMPLES_PER_TEACHER = config.SAMPLES_PER_TEACHER

# Tracking state for live attendance and smart attendance (Item 4, 5, 6, 7)
LIVE_ATTENDANCE_TRACKER = {}
SMART_ATTENDANCE_TRACKER = {}

# LBPH prediction "distance" — lower means more confident match.
# Anything above this is treated as an unrecognized/unknown face.
RECOGNITION_CONFIDENCE_THRESHOLD = 75

# NOTE: There used to be an auto-fallback here — if a period started and no
# teacher was face-confirmed present within AUTO_ABSENT_GRACE_MINUTES, the
# period would auto-open (teacher marked Absent) and student attendance
# would unlock anyway. That fallback has been removed by design: student
# attendance must NEVER be marked for a period unless a teacher was actually
# face-recognized present for it. See ensure_auto_attendance() below, which
# is now a no-op kept only so existing call sites don't need to change.
AUTO_ABSENT_GRACE_MINUTES = 30

# Default skeleton timetable used by "Generate Default Day" (matches the
# common 8:20am - 4:00pm school day with a tea break and a lunch break)
DEFAULT_DAY_START = "08:20"
DEFAULT_DAY_END = "16:00"
DEFAULT_PERIOD_MINUTES = 50
DEFAULT_TEA_BREAK = ("10:20", "10:40")
DEFAULT_LUNCH_BREAK = ("12:40", "13:20")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = None          # student recognizer — loaded after training, see load_recognizer()
teacher_recognizer = None  # teacher recognizer — loaded after training, see load_teacher_recognizer()

# In-memory per-session tracking state for Exam Hall Malpractice Detection,
# keyed by exam_sessions.id. Not persisted — fine to lose on server restart,
# a restart just means alert cooldown timers reset.
EXAM_STATE = {}
EXAM_ROSTER_CACHE = {}  # exam_session_id -> {student_id: sqlite3.Row} of allocated seats

# Where exam malpractice evidence photos are saved (served as static files).
EXAM_PHOTOS_DIR = os.path.join(BASE_DIR, "static", "exam_photos")

# Same idea for Class Activeness Monitoring. ACTIVENESS_CLASS_ROSTER caches
# {label(student_id): (name, roll_no)} per session so the periodic
# identify_face_fn callback doesn't hit the DB on every resolve -- it's built
# once at activeness_start() from the selected class's students.
ACTIVENESS_STATE = {}
ACTIVENESS_CLASS_ROSTER = {}

# Where classroom-activeness evidence photos are saved (served as static
# files) -- same pattern as EXAM_PHOTOS_DIR above.
ACTIVENESS_PHOTOS_DIR = os.path.join(BASE_DIR, "static", "activeness_photos")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key-in-production")

from flask_cors import CORS
from flask_sock import Sock
import api_routes

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
sock = Sock(app)
api_routes.init_api(app, sock)


# ---------------------------------------------------------------------
# Static Evidence Serving Routes
# Maps frontend requests (/activeness_photos/..., /exam_photos/...)
# directly to static/activeness_photos and static/exam_photos
# with path traversal protection and Evidence vault fallback.
# ---------------------------------------------------------------------
@app.route("/activeness_photos/<path:filename>")
def serve_activeness_photo(filename):
    safe_path = os.path.normpath(filename).replace("\\", "/")
    if safe_path.startswith("../") or safe_path.startswith("/") or ".." in safe_path.split("/"):
        abort(404)

    # 1. Primary location: static/activeness_photos/<filename>
    full_path = os.path.join(ACTIVENESS_PHOTOS_DIR, safe_path)
    if os.path.isfile(full_path):
        directory = os.path.dirname(full_path)
        base_name = os.path.basename(full_path)
        return send_from_directory(directory, base_name)

    # 2. Fallback: Check Evidence vault in case file was archived
    parts = safe_path.split("/")
    if len(parts) >= 2:
        event_folder_name = parts[-2]
        file_name = parts[-1]
        event_num_str = event_folder_name.replace("event_", "")
        evidence_dir = os.path.join(BASE_DIR, "Evidence")
        if os.path.isdir(evidence_dir):
            for root, dirs, files in os.walk(evidence_dir):
                if file_name in files and event_num_str in os.path.basename(root):
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    shutil.copy2(os.path.join(root, file_name), full_path)
                    return send_from_directory(os.path.dirname(full_path), file_name)

    # 3. Genuinely does not exist
    abort(404)


@app.route("/exam_photos/<path:filename>")
def serve_exam_photo(filename):
    safe_path = os.path.normpath(filename).replace("\\", "/")
    if safe_path.startswith("../") or safe_path.startswith("/") or ".." in safe_path.split("/"):
        abort(404)

    full_path = os.path.join(EXAM_PHOTOS_DIR, safe_path)
    if os.path.isfile(full_path):
        directory = os.path.dirname(full_path)
        base_name = os.path.basename(full_path)
        return send_from_directory(directory, base_name)

    parts = safe_path.split("/")
    if len(parts) >= 2:
        event_folder_name = parts[-2]
        file_name = parts[-1]
        event_num_str = event_folder_name.replace("event_", "")
        evidence_dir = os.path.join(BASE_DIR, "Evidence")
        if os.path.isdir(evidence_dir):
            for root, dirs, files in os.walk(evidence_dir):
                if file_name in files and event_num_str in os.path.basename(root):
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    shutil.copy2(os.path.join(root, file_name), full_path)
                    return send_from_directory(os.path.dirname(full_path), file_name)

    abort(404)


# ---------------------------------------------------------------------
# React Single Page Application (SPA) Serving
# ---------------------------------------------------------------------
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

@app.route("/app")
@app.route("/app/")
def serve_app_index():
    if os.path.isdir(FRONTEND_DIST):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return "Frontend bundle not found. Please run 'npm run build' in frontend/.", 503

@app.route("/app/assets/<path:filename>")
def serve_app_assets(filename):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        return send_from_directory(assets_dir, filename)
    abort(404)


# ---------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
    except Exception:
        pass
    if has_request_context():
        if not hasattr(g, "_opened_connections"):
            g._opened_connections = []
        g._opened_connections.append(conn)
    return conn


@app.teardown_appcontext
def close_db(exception=None):
    connections = getattr(g, "_opened_connections", None)
    if connections:
        for conn in list(connections):
            try:
                conn.close()
            except Exception:
                pass
        g._opened_connections = []


@contextmanager
def get_db_transaction():
    """
    Context manager for atomic write operations.
    Begins transaction, yields connection, commits on success,
    rolls back on error, and guarantees connection closure.
    """
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if has_request_context() and hasattr(g, "_opened_connections"):
            try:
                g._opened_connections.remove(conn)
            except (ValueError, AttributeError):
                pass


def execute_with_retry(func, max_retries=5, initial_delay=0.05):
    """
    Executes a database callable with exponential backoff retry for transient
    sqlite3.OperationalError: database is locked / database is busy.
    """
    delay = initial_delay
    last_err = None
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if "locked" in err_msg or "busy" in err_msg:
                last_err = e
                time.sleep(delay + random.uniform(0.01, 0.05))
                delay *= 2
            else:
                raise
    raise last_err



def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            start_time TEXT,
            end_time TEXT,
            late_after_minutes INTEGER NOT NULL DEFAULT 10
        )
    """)

    # Migration: add the schedule columns if this DB was created before they existed
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(classes)").fetchall()}
    if "start_time" not in existing_cols:
        cur.execute("ALTER TABLE classes ADD COLUMN start_time TEXT")
    if "end_time" not in existing_cols:
        cur.execute("ALTER TABLE classes ADD COLUMN end_time TEXT")
    if "late_after_minutes" not in existing_cols:
        cur.execute("ALTER TABLE classes ADD COLUMN late_after_minutes INTEGER NOT NULL DEFAULT 10")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
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

    # Migration: status fields used by Automatic Data Expiry (9.4) --
    # lets an admin mark a student graduated/inactive, which (after a
    # grace period) makes their face photos eligible for deletion.
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(students)").fetchall()}
    if "status" not in existing_cols:
        cur.execute("ALTER TABLE students ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
    if "status_set_at" not in existing_cols:
        cur.execute("ALTER TABLE students ADD COLUMN status_set_at TEXT")
    if "face_deleted_at" not in existing_cols:
        cur.execute("ALTER TABLE students ADD COLUMN face_deleted_at TEXT")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            att_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('Present','Absent','Late')),
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
            UNIQUE(student_id, att_date)
        )
    """)

    # --- Teachers ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT
        )
    """)

    # --- Timetable: the fixed daily period schedule for a class ---
    # period_type: 'class' (a taught period) or 'break' (tea/lunch, no attendance)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS timetable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            period_number INTEGER NOT NULL,
            period_type TEXT NOT NULL DEFAULT 'class' CHECK(period_type IN ('class','break')),
            subject TEXT,
            teacher_id INTEGER,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
            FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE SET NULL
        )
    """)

    # --- Period sessions: one row per (class, timetable period, date) once a
    #     teacher has been recognized/confirmed and the period "goes live" ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS period_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            timetable_id INTEGER NOT NULL,
            session_date TEXT NOT NULL,
            teacher_id INTEGER,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            teacher_status TEXT NOT NULL DEFAULT 'present',
            auto_marked INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
            FOREIGN KEY (timetable_id) REFERENCES timetable (id) ON DELETE CASCADE,
            FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE SET NULL,
            UNIQUE(timetable_id, session_date)
        )
    """)

    # --- Migration for DBs created before teacher_status/auto_marked existed ---
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(period_sessions)")}
    if "teacher_status" not in existing_cols:
        cur.execute("ALTER TABLE period_sessions ADD COLUMN teacher_status TEXT NOT NULL DEFAULT 'present'")
    if "auto_marked" not in existing_cols:
        cur.execute("ALTER TABLE period_sessions ADD COLUMN auto_marked INTEGER NOT NULL DEFAULT 0")

    # --- Period attendance: student check-ins for one specific live session ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS period_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            marked_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES period_sessions (id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
            UNIQUE(session_id, student_id)
        )
    """)

    # --- Exam Hall Malpractice Detection ---
    # Rooms an exam can be allocated to.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exam_rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_name TEXT UNIQUE NOT NULL,
            capacity INTEGER NOT NULL DEFAULT 30,
            location TEXT
        )
    """)

    # One exam sitting in one room. Students get allocated to it (by USN)
    # before monitoring starts, so every camera identification can be
    # checked against a known, closed roster for that room instead of the
    # whole school.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exam_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_name TEXT NOT NULL,
            room_id INTEGER NOT NULL,
            class_id INTEGER,
            started_at TEXT,
            ended_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (room_id) REFERENCES exam_rooms (id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE SET NULL
        )
    """)

    # Seat allocation: which student (identified by USN/roll_no) sits where
    # for this exam session.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exam_seat_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            seat_no TEXT,
            FOREIGN KEY (exam_session_id) REFERENCES exam_sessions (id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
            UNIQUE(exam_session_id, student_id)
        )
    """)

    # Every malpractice alert raised during a monitored exam session,
    # attributed (where the face could be matched against the allocated
    # roster) to a specific student/USN, with a saved photo as proof.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exam_malpractice_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_session_id INTEGER NOT NULL,
            student_id INTEGER,
            usn TEXT,
            student_name TEXT,
            seat_no TEXT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'low',
            detail TEXT,
            confidence REAL,
            photo_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (exam_session_id) REFERENCES exam_sessions (id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE SET NULL
        )
    """)

    # Migration: a short buffered video clip alongside the evidence photo,
    # for medium/high-severity alerts (looking_back / phone_use / copying)
    # -- see exam_ai.py's VIDEO CLIP EVIDENCE notes and needs_video_clip.
    # Recorded client-side and uploaded separately, same pattern as
    # activeness_alerts.video_path below.
    exam_alert_cols = {row["name"] for row in cur.execute("PRAGMA table_info(exam_malpractice_alerts)").fetchall()}
    if "video_path" not in exam_alert_cols:
        cur.execute("ALTER TABLE exam_malpractice_alerts ADD COLUMN video_path TEXT")

    # Migration: drop the older, un-attributed classroom "Malpractice
    # Monitor" concept if it exists from a previous version of this app —
    # replaced entirely by the exam-hall/USN-attributed feature above.
    cur.execute("DROP TABLE IF EXISTS malpractice_alerts")
    cur.execute("DROP TABLE IF EXISTS malpractice_sessions")

    # Migration: the old Active Participation Tracking feature (hand-raise
    # counting) has been replaced entirely by Class Activeness Monitoring
    # below -- drop its tables rather than leave dead data around.
    cur.execute("DROP TABLE IF EXISTS participation_snapshots")
    cur.execute("DROP TABLE IF EXISTS participation_sessions")

    # --- Class Activeness Monitoring ---
    # A monitored classroom session (webcam pointed at the room during a
    # lesson). This is a teacher-facing engagement AID, not a disciplinary
    # record -- see activeness_ai.py for the accuracy caveats that apply
    # to every number this feature produces.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activeness_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            label TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE
        )
    """)

    # One row per tracked face per session, kept up to date (not an event
    # log) -- a live "cumulative sample counts per student" table, upserted
    # every analyzed frame. student_id/name are NULL until the periodic
    # recognizer check in activeness_ai resolves an identity. The *_samples
    # columns are running counts of scored checks in each band, which is
    # what the session report turns into a percentage -- e.g. "active in
    # 43 of 50 checks (86%)". These are sample-based estimates, not a
    # measurement of the full class duration.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activeness_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            student_id INTEGER,
            student_name TEXT,
            band TEXT,
            total_scored INTEGER NOT NULL DEFAULT 0,
            active_samples INTEGER NOT NULL DEFAULT 0,
            turned_away_samples INTEGER NOT NULL DEFAULT 0,
            eyes_closed_samples INTEGER NOT NULL DEFAULT 0,
            phone_use_samples INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES activeness_sessions (id) ON DELETE CASCADE,
            UNIQUE(session_id, track_id)
        )
    """)

    # Discrete evidence photos, saved the first time a disengagement
    # signal (turned_away / eyes_closed / phone_use) is confirmed for a
    # student, with a cooldown so it isn't a flood of near-identical
    # shots. Same "photo as supporting context for a human to review"
    # pattern as exam_malpractice_alerts -- NOT proof on its own; every
    # row here is a heuristic signal, always labelled as such in detail.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activeness_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER,
            student_name TEXT,
            alert_type TEXT NOT NULL,
            detail TEXT,
            photo_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES activeness_sessions (id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE SET NULL
        )
    """)

    # Migration: severity + a short buffered video clip for the escalated
    # "sleeping" / "phone_use_sustained" bands (activeness_ai.py). The clip
    # is recorded client-side (the server only ever sees single JPEG
    # frames) and uploaded separately once the browser learns an alert
    # needs one -- see activeness_alert_video() below.
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(activeness_alerts)").fetchall()}
    if "severity" not in existing_cols:
        cur.execute("ALTER TABLE activeness_alerts ADD COLUMN severity TEXT NOT NULL DEFAULT 'low'")
    if "video_path" not in existing_cols:
        cur.execute("ALTER TABLE activeness_alerts ADD COLUMN video_path TEXT")

    # --- Discipline flags (feed data for Behavioral Risk Prediction, 9.1) ---
    # A teacher-logged incident note. Deliberately NOT auto-generated from
    # camera footage identifying a student -- the malpractice camera can't
    # reliably say *which* student did what, so this stays a human-entered,
    # attributable record. The AI part is the scoring/trend analysis on
    # top of these records, done in behavior_ai.py.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS discipline_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            flag_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'low' CHECK(severity IN ('low','medium','high')),
            note TEXT,
            created_by TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE
        )
    """)

    # --- Data Expiry audit log (Advanced AI Feature 9.4) ---
    # Records every face-photo purge, automatic or manual, so deletions
    # stay auditable even though the photos themselves are gone.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS data_expiry_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            class_name TEXT,
            reason TEXT NOT NULL,
            detail TEXT,
            photos_removed INTEGER NOT NULL DEFAULT 0,
            triggered_by TEXT NOT NULL DEFAULT 'automatic',
            deleted_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE SET NULL
        )
    """)

    # --- Enhanced Face Embeddings & Evidence Tracking Tables (Item 19) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_type TEXT NOT NULL,
            person_id INTEGER NOT NULL,
            pose_tag TEXT NOT NULL DEFAULT 'FRONT',
            quality_score REAL,
            embedding_blob BLOB,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings ON face_embeddings(person_type, person_id)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            track_id INTEGER,
            person_type TEXT,
            person_id INTEGER,
            usn TEXT,
            confidence REAL,
            quality_score REAL,
            timestamp TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            student_id INTEGER,
            usn TEXT,
            current_activity TEXT,
            seat_no TEXT,
            status TEXT,
            last_seen TEXT NOT NULL
        )
    """)

    # Migration: identity_source on exam_events
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exam_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_session_id INTEGER NOT NULL,
            track_id INTEGER,
            student_id INTEGER,
            usn TEXT,
            student_name TEXT,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'CONFIRMED_EVENT',
            severity TEXT NOT NULL DEFAULT 'medium',
            duration_seconds REAL,
            confidence REAL,
            photo_path TEXT,
            video_path TEXT,
            identity_source TEXT DEFAULT 'FACE_VERIFIED',
            created_at TEXT NOT NULL
        )
    """)
    existing_event_cols = {row["name"] for row in cur.execute("PRAGMA table_info(exam_events)").fetchall()}
    if "identity_source" not in existing_event_cols:
        cur.execute("ALTER TABLE exam_events ADD COLUMN identity_source TEXT DEFAULT 'FACE_VERIFIED'")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS _deprecated_exam_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_session_id INTEGER NOT NULL,
            track_id INTEGER,
            student_id INTEGER,
            usn TEXT,
            student_name TEXT,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'CONFIRMED_EVENT',
            severity TEXT NOT NULL DEFAULT 'medium',
            duration_seconds REAL,
            confidence REAL,
            photo_path TEXT,
            video_path TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evidence_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Create default admin user if none exists
    cur.execute("SELECT COUNT(*) AS c FROM users")
    if cur.fetchone()["c"] == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin")
        )

    # Seed one default class if none exists
    cur.execute("SELECT COUNT(*) AS c FROM classes")
    if cur.fetchone()["c"] == 0:
        cur.execute("INSERT INTO classes (name) VALUES (?)", ("Class A",))

    conn.commit()
    conn.close()


def ensure_face_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TEACHER_DATASET_DIR, exist_ok=True)
    os.makedirs(EXAM_PHOTOS_DIR, exist_ok=True)
    os.makedirs(ACTIVENESS_PHOTOS_DIR, exist_ok=True)
    os.makedirs(config.EVIDENCE_VAULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(FACE_DIR, "temp"), exist_ok=True)


def load_recognizer():
    """(Re)load the trained student LBPH model from disk into the global `recognizer`.
    If the file exists but is empty/corrupt (e.g. an interrupted training run), this
    logs a warning and leaves `recognizer` as None instead of crashing the app."""
    global recognizer
    if os.path.exists(TRAINER_FILE) and os.path.getsize(TRAINER_FILE) > 0:
        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(TRAINER_FILE)
            recognizer = model
            return True
        except cv2.error:
            print(f"[warning] {TRAINER_FILE} exists but could not be read (corrupt/empty). "
                  f"Re-train the student face model on the Face Setup page.")
    recognizer = None
    return False


def load_teacher_recognizer():
    """(Re)load the trained teacher LBPH model from disk into `teacher_recognizer`.
    If the file exists but is empty/corrupt (e.g. an interrupted training run), this
    logs a warning and leaves `teacher_recognizer` as None instead of crashing the app."""
    global teacher_recognizer
    if os.path.exists(TEACHER_TRAINER_FILE) and os.path.getsize(TEACHER_TRAINER_FILE) > 0:
        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(TEACHER_TRAINER_FILE)
            teacher_recognizer = model
            return True
        except cv2.error:
            print(f"[warning] {TEACHER_TRAINER_FILE} exists but could not be read (corrupt/empty). "
                  f"Re-train the teacher face model on the Teachers page.")
    teacher_recognizer = None
    return False


def decode_base64_image(data_url):
    """Turn a 'data:image/jpeg;base64,....' string (from a <canvas>) into a cv2 BGR image."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        binary = base64.b64decode(data_url)
        arr = np.frombuffer(binary, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def detect_faces(gray_img):
    """Return a list of (x, y, w, h) boxes for faces found in a grayscale image."""
    return face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )


# ---------------------------------------------------------------------
# Automatic Data Expiry (Advanced AI Feature 9.4) — helpers
# ---------------------------------------------------------------------
def purge_student_face_data(conn, candidate, triggered_by="automatic"):
    """Delete one student's enrolled face photos and write an audit log
    row. Deliberately does NOT touch attendance/discipline records or
    the student row itself — see data_expiry.py for the rationale.
    Does not retrain the recognition model (that's a manual "Train
    Model" step on Face Setup, same as after any other face deletion)."""
    student_dir = os.path.join(DATASET_DIR, str(candidate["student_id"]))
    photo_count = face_sample_count(candidate["student_id"])

    if os.path.isdir(student_dir):
        shutil.rmtree(student_dir)

    now_iso = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE students SET face_deleted_at = ? WHERE id = ?",
        (now_iso, candidate["student_id"]),
    )
    conn.execute(
        "INSERT INTO data_expiry_log "
        "(student_id, roll_no, name, class_name, reason, detail, photos_removed, triggered_by, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (candidate["student_id"], candidate["roll_no"], candidate["name"], candidate["class_name"],
         candidate["reason"], candidate["detail"], photo_count, triggered_by, now_iso),
    )
    return photo_count


def run_data_expiry_sweep(conn, triggered_by="automatic"):
    """Find every student currently due for face-data expiry and purge
    them. Cheap to call opportunistically (like ensure_auto_attendance)
    — it's just a DB query plus an os.listdir per student, and does
    nothing at all once nobody is due. Returns the list purged."""
    due = data_expiry.find_due_for_expiry(
        conn, has_face_data_fn=lambda sid: face_sample_count(sid) > 0
    )
    for candidate in due:
        purge_student_face_data(conn, candidate, triggered_by=triggered_by)
    if due:
        conn.commit()
    return due


def better_teacher_match(face_img, student_confidence, conn):
    """Cross-check a detected face against the teacher model as well as the
    student model, so a teacher standing in front of the camera doesn't get
    misidentified and marked present as a student.

    LBPH's predict() always returns *some* label — even for a face it has
    never seen — so a low confidence value on its own doesn't guarantee the
    face actually belongs to that class. Comparing both models and trusting
    whichever produced the lower (more confident) distance is a much more
    accurate way to decide whether a face is a teacher's or a student's.

    Returns the matching teacher row if the teacher model is a confident
    AND better match than the student model, otherwise None.
    """
    if teacher_recognizer is None:
        return None

    teacher_label, teacher_confidence = teacher_recognizer.predict(face_img)

    is_confident = teacher_confidence <= RECOGNITION_CONFIDENCE_THRESHOLD
    is_better_than_student = (
        student_confidence > RECOGNITION_CONFIDENCE_THRESHOLD
        or teacher_confidence < student_confidence
    )

    if is_confident and is_better_than_student:
        teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_label,)).fetchone()
        if teacher:
            return teacher, teacher_confidence
    return None


def face_sample_count(person_id, base_dir=DATASET_DIR):
    person_dir = os.path.join(base_dir, str(person_id))
    if not os.path.isdir(person_dir):
        return 0
    return len(glob.glob(os.path.join(person_dir, "*.jpg")))


def determine_live_status(cls_row):
    """
    Decide Present vs Late for a check-in happening right now, based on the
    class's scheduled start_time ('HH:MM') and its late_after_minutes grace
    period. Classes with no start_time set always count as Present.
    """
    start_time = cls_row["start_time"] if cls_row else None
    if not start_time:
        return "Present"
    try:
        hour, minute = map(int, start_time.split(":"))
    except (ValueError, AttributeError):
        return "Present"

    today = date.today()
    start_dt = datetime(today.year, today.month, today.day, hour, minute)
    grace = cls_row["late_after_minutes"] if cls_row["late_after_minutes"] is not None else 10
    late_cutoff = start_dt + timedelta(minutes=grace)

    return "Late" if datetime.now() > late_cutoff else "Present"


def generate_default_timetable(conn, class_id):
    """
    Build a skeleton day for a class: back-to-back periods from
    DEFAULT_DAY_START to DEFAULT_DAY_END, with a tea break and lunch break
    carved out at fixed times. Subject/teacher are left blank for the admin
    to fill in per period afterwards. Returns the number of rows created.
    """
    def to_minutes(hhmm):
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m

    def to_hhmm(total_minutes):
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

    day_start = to_minutes(DEFAULT_DAY_START)
    day_end = to_minutes(DEFAULT_DAY_END)
    tea_start, tea_end = to_minutes(DEFAULT_TEA_BREAK[0]), to_minutes(DEFAULT_TEA_BREAK[1])
    lunch_start, lunch_end = to_minutes(DEFAULT_LUNCH_BREAK[0]), to_minutes(DEFAULT_LUNCH_BREAK[1])

    blocks = []  # (start, end, type)
    cursor = day_start
    period_num = 1
    while cursor < day_end:
        # Insert breaks exactly at their fixed times
        if cursor == tea_start:
            blocks.append((tea_start, tea_end, "break", "Tea Break"))
            cursor = tea_end
            continue
        if cursor == lunch_start:
            blocks.append((lunch_start, lunch_end, "break", "Lunch Break"))
            cursor = lunch_end
            continue

        # Otherwise a normal class period, but don't run past the next break/day-end
        next_stop = day_end
        for b_start in (tea_start, lunch_start):
            if cursor < b_start < next_stop:
                next_stop = b_start
        period_end = min(cursor + DEFAULT_PERIOD_MINUTES, next_stop)
        if period_end <= cursor:
            break
        blocks.append((cursor, period_end, "class", None))
        cursor = period_end

    cur = conn.cursor()
    cur.execute("DELETE FROM timetable WHERE class_id = ?", (class_id,))
    period_num = 1
    for start, end, btype, label in blocks:
        cur.execute("""
            INSERT INTO timetable (class_id, period_number, period_type, subject, teacher_id, start_time, end_time)
            VALUES (?, ?, ?, ?, NULL, ?, ?)
        """, (class_id, period_num, btype, label, to_hhmm(start), to_hhmm(end)))
        period_num += 1
    conn.commit()
    return len(blocks)


def get_current_period(conn, class_id, now=None):
    """
    Return the timetable row (as a dict) covering the current time for this
    class, or None if outside all scheduled periods (e.g. before/after the
    school day). Works for both 'class' and 'break' rows.
    """
    now = now or datetime.now()
    now_str = now.strftime("%H:%M")
    row = conn.execute("""
        SELECT * FROM timetable
        WHERE class_id = ? AND start_time <= ? AND end_time > ?
        ORDER BY start_time
        LIMIT 1
    """, (class_id, now_str, now_str)).fetchone()
    return dict(row) if row else None


def get_period_timing(period, now=None):
    """
    Given a timetable row dict and an optional datetime `now`, compute authoritative
    timestamps based strictly on the scheduled period start_time.
    Returns:
      period_start_dt: datetime of scheduled start today
      period_end_dt: datetime of scheduled end today
      grace_deadline_dt: period_start_dt + timedelta(minutes=AUTO_ABSENT_GRACE_MINUTES)
      period_start_str: 'HH:MM'
      period_end_str: 'HH:MM'
      grace_deadline_str: 'HH:MM'
      remaining_seconds: seconds remaining until grace deadline (0 if elapsed)
      is_past_grace: True if now >= grace_deadline_dt
      elapsed_seconds: seconds since scheduled start
    """
    now = now or datetime.now()
    if not period:
        return None
    if hasattr(period, "keys") and not isinstance(period, dict):
        try:
            period = dict(period)
        except Exception:
            return None
    elif not isinstance(period, dict):
        return None
    if not period.get("start_time") or not period.get("end_time"):
        return None
    try:
        sh, sm = map(int, period["start_time"].split(":"))
        eh, em = map(int, period["end_time"].split(":"))
    except (ValueError, AttributeError, KeyError):
        return None

    today = now.date()
    period_start_dt = datetime(today.year, today.month, today.day, sh, sm, 0)
    period_end_dt = datetime(today.year, today.month, today.day, eh, em, 0)
    if period_end_dt < period_start_dt:
        period_end_dt += timedelta(days=1)

    grace_deadline_dt = period_start_dt + timedelta(minutes=AUTO_ABSENT_GRACE_MINUTES)
    rem = int((grace_deadline_dt - now).total_seconds())
    remaining_seconds = max(0, rem)
    is_past_grace = now >= grace_deadline_dt
    elapsed = int((now - period_start_dt).total_seconds())

    return {
        "period_start_dt": period_start_dt,
        "period_end_dt": period_end_dt,
        "grace_deadline_dt": grace_deadline_dt,
        "period_start_str": period["start_time"],
        "period_end_str": period["end_time"],
        "grace_deadline_str": grace_deadline_dt.strftime("%H:%M"),
        "remaining_seconds": remaining_seconds,
        "is_past_grace": is_past_grace,
        "elapsed_seconds": max(0, elapsed),
    }


def get_active_teacher_session(conn, class_id, now=None):
    """
    Return the active period session for this class if student attendance is allowed:
    1) A teacher has been face-confirmed present (teacher_id IS NOT NULL), OR
    2) The 30-minute grace period has passed without a teacher and the automatic student
       attendance policy has activated (teacher_status='absent', auto_marked=1).

    Returns None if:
    - Outside class periods (or during break)
    - Within the 30-minute grace period and no teacher confirmed yet (WAITING_FOR_TEACHER)
    - Period session has ended (ended_at IS NOT NULL)
    """
    now = now or datetime.now()
    period = get_current_period(conn, class_id, now=now)
    if not period or period["period_type"] != "class":
        return None

    # Trigger auto-attendance if 30 minutes have elapsed
    ensure_auto_attendance(conn, class_id, now=now)

    today = now.date().isoformat()
    row = conn.execute("""
        SELECT ps.*, t.name AS teacher_name, tt.subject
        FROM period_sessions ps
        JOIN timetable tt ON tt.id = ps.timetable_id
        LEFT JOIN teachers t ON t.id = ps.teacher_id
        WHERE ps.timetable_id = ? AND ps.session_date = ? AND ps.ended_at IS NULL
    """, (period["id"], today)).fetchone()

    if not row:
        return None

    if row["teacher_id"] is not None:
        return dict(row)

    timing = get_period_timing(period, now=now)
    if timing and timing["is_past_grace"] and row["teacher_status"] == "absent":
        return dict(row)

    return None


def ensure_auto_attendance(conn, class_id, now=None):
    """
    Server-authoritative 30-minute teacher absent auto-attendance rule.
    Calculates: scheduled_period_start + 30 minutes = teacher_grace_deadline.

    If the scheduled period is active, no teacher has been confirmed, and
    now >= teacher_grace_deadline:
      1. Sets teacher status to ABSENT in database (period_sessions with teacher_id=NULL,
         teacher_status='absent', auto_marked=1).
      2. Activates the existing automatic student-attendance policy: opens the session
         so student faces can be scanned and marked present for the period.

    If now < teacher_grace_deadline:
      - Does nothing. Session remains in WAITING_FOR_TEACHER.

    Returns the period_sessions row dict if active/created, else None.
    """
    now = now or datetime.now()
    period = get_current_period(conn, class_id, now=now)
    if not period or period["period_type"] != "class":
        return None

    timing = get_period_timing(period, now=now)
    if not timing:
        return None

    today_str = now.date().isoformat()
    existing = conn.execute(
        "SELECT * FROM period_sessions WHERE timetable_id = ? AND session_date = ?",
        (period["id"], today_str)
    ).fetchone()

    if existing:
        return dict(existing)

    # Within the 30-minute grace window — teacher can still arrive on time
    if not timing["is_past_grace"]:
        return None

    # At or after 30 minutes without teacher: auto-open session as teacher absent
    started_at_iso = timing["period_start_dt"].isoformat(timespec="seconds")
    conn.execute("""
        INSERT INTO period_sessions
            (class_id, timetable_id, session_date, teacher_id, started_at, teacher_status, auto_marked)
        VALUES (?, ?, ?, NULL, ?, 'absent', 1)
        ON CONFLICT(timetable_id, session_date) DO NOTHING
    """, (class_id, period["id"], today_str, started_at_iso))
    conn.commit()

    session_row = conn.execute(
        "SELECT * FROM period_sessions WHERE timetable_id = ? AND session_date = ?",
        (period["id"], today_str)
    ).fetchone()
    return dict(session_row) if session_row else None


def list_active_teacher_sessions(conn):
    """
    Every period session currently in progress (a teacher has been
    face-confirmed present and the period hasn't ended) across all classes,
    for the 'Teachers Present Now' panels on the dashboard and Smart
    Attendance page. Includes a live present/total student count per class.
    """
    today = date.today().isoformat()
    rows = conn.execute("""
        SELECT ps.id AS session_id, c.id AS class_id, c.name AS class_name,
               t.name AS teacher_name, tt.subject, tt.start_time, tt.end_time,
               ps.started_at, ps.teacher_status, ps.auto_marked,
               (SELECT COUNT(*) FROM period_attendance pa WHERE pa.session_id = ps.id) AS present_count,
               (SELECT COUNT(*) FROM students s WHERE s.class_id = ps.class_id) AS total_students
        FROM period_sessions ps
        JOIN classes c ON c.id = ps.class_id
        JOIN timetable tt ON tt.id = ps.timetable_id
        LEFT JOIN teachers t ON t.id = ps.teacher_id
        WHERE ps.session_date = ? AND ps.ended_at IS NULL
        ORDER BY tt.start_time
    """, (today,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Welcome back, " + user["username"] + "!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?", (session["user_id"],)
        ).fetchone()

        if not user or not check_password_hash(user["password_hash"], current):
            flash("Current password is incorrect.", "danger")
        elif new != confirm:
            flash("New passwords do not match.", "danger")
        elif len(new) < 4:
            flash("New password must be at least 4 characters.", "danger")
        else:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(new), user["id"])
            )
            conn.commit()
            flash("Password updated successfully.", "success")
            conn.close()
            return redirect(url_for("dashboard"))
        conn.close()

    return render_template("change_password.html")


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()

    # Sweep every class so periods whose grace period has elapsed with no
    # teacher confirmed get auto-opened (teacher Absent, students Present)
    # even if nobody has the Smart Attendance page open right now.
    for cls in conn.execute("SELECT id FROM classes").fetchall():
        ensure_auto_attendance(conn, cls["id"])

    # Automatic Data Expiry (9.4) — cheap sweep, purges face photos for
    # anyone currently due (graduated/inactive past grace period, or
    # long-inactive). See data_expiry.py.
    expired_now = run_data_expiry_sweep(conn, triggered_by="automatic")
    if expired_now:
        names = ", ".join(c["name"] for c in expired_now[:3])
        extra = f" and {len(expired_now) - 3} more" if len(expired_now) > 3 else ""
        flash(f"Auto-deleted face photos for {len(expired_now)} student(s) ({names}{extra}) — retention policy.", "info")

    total_students = conn.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    total_classes = conn.execute("SELECT COUNT(*) c FROM classes").fetchone()["c"]

    today = date.today().isoformat()
    present_today = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE att_date = ? AND status = 'Present'",
        (today,)
    ).fetchone()["c"]
    absent_today = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE att_date = ? AND status = 'Absent'",
        (today,)
    ).fetchone()["c"]
    late_today = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE att_date = ? AND status = 'Late'",
        (today,)
    ).fetchone()["c"]

    recent = conn.execute("""
        SELECT a.att_date, a.status, s.name, s.roll_no, c.name AS class_name
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        JOIN classes c ON c.id = a.class_id
        ORDER BY a.id DESC
        LIMIT 8
    """).fetchall()

    teacher_sessions = list_active_teacher_sessions(conn)

    # Behavioral risk snapshot (Advanced AI Feature 9.1) — only bother
    # scoring students if any incidents have ever been logged, so this
    # stays a no-op cost on a fresh install.
    high_risk_count = 0
    has_discipline_flags = conn.execute("SELECT COUNT(*) c FROM discipline_flags").fetchone()["c"] > 0
    if has_discipline_flags:
        for cls in conn.execute("SELECT id FROM classes").fetchall():
            for r in behavior_ai.compute_class_risk(conn, cls["id"]):
                if r["level"] == "High":
                    high_risk_count += 1

    # Class Activeness Monitoring -- cheap today-only snapshot for the
    # dashboard card: average "active" percentage across every student
    # scored in sessions started today, and whether a session is running.
    activeness_row = conn.execute(
        "SELECT COALESCE(SUM(sn.active_samples), 0) AS active_sum, "
        "COALESCE(SUM(sn.total_scored), 0) AS scored_sum "
        "FROM activeness_snapshots sn "
        "JOIN activeness_sessions acs ON acs.id = sn.session_id "
        "WHERE date(acs.started_at) = ?",
        (today,)
    ).fetchone()
    activeness_avg_pct_today = (
        round(100 * activeness_row["active_sum"] / activeness_row["scored_sum"])
        if activeness_row["scored_sum"] else None
    )
    activeness_live_now = conn.execute(
        "SELECT COUNT(*) c FROM activeness_sessions WHERE ended_at IS NULL"
    ).fetchone()["c"] > 0

    recent_exam_events = conn.execute("""
        SELECT a.*, r.room_name, es.exam_name
        FROM exam_malpractice_alerts a
        JOIN exam_sessions es ON es.id = a.exam_session_id
        JOIN exam_rooms r ON r.id = es.room_id
        ORDER BY a.id DESC LIMIT 6
    """).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_students=total_students,
        total_classes=total_classes,
        present_today=present_today,
        absent_today=absent_today,
        late_today=late_today,
        recent=recent,
        today=today,
        teacher_sessions=teacher_sessions,
        auto_absent_grace_minutes=AUTO_ABSENT_GRACE_MINUTES,
        high_risk_count=high_risk_count,
        activeness_avg_pct_today=activeness_avg_pct_today,
        activeness_live_now=activeness_live_now,
        recent_exam_events=recent_exam_events,
    )


@app.route("/api/teacher-sessions")
@login_required
def api_teacher_sessions():
    """Live list of teachers currently present (JSON), used to auto-refresh
    the 'Teachers Present Now' panel without a full page reload. Also sweeps
    every class for periods whose grace period has elapsed with no teacher
    confirmed, auto-opening them (see ensure_auto_attendance)."""
    conn = get_db()
    for cls in conn.execute("SELECT id FROM classes").fetchall():
        ensure_auto_attendance(conn, cls["id"])
    sessions = list_active_teacher_sessions(conn)
    conn.close()
    return jsonify(sessions)


# ---------------------------------------------------------------------
# Class management
# ---------------------------------------------------------------------
@app.route("/classes", methods=["GET", "POST"])
@login_required
def classes():
    conn = get_db()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        start_time = request.form.get("start_time", "").strip() or None
        end_time = request.form.get("end_time", "").strip() or None
        late_after = request.form.get("late_after_minutes", "10").strip() or "10"
        if name:
            try:
                conn.execute(
                    "INSERT INTO classes (name, start_time, end_time, late_after_minutes) VALUES (?, ?, ?, ?)",
                    (name, start_time, end_time, int(late_after))
                )
                conn.commit()
                flash(f'Class "{name}" added.', "success")
            except sqlite3.IntegrityError:
                flash("That class already exists.", "danger")
        return redirect(url_for("classes"))

    all_classes = conn.execute("""
        SELECT c.id, c.name, c.start_time, c.end_time, c.late_after_minutes,
               COUNT(s.id) AS student_count
        FROM classes c
        LEFT JOIN students s ON s.class_id = c.id
        GROUP BY c.id
        ORDER BY c.name
    """).fetchall()
    conn.close()
    return render_template("classes.html", classes=all_classes)


@app.route("/classes/<int:class_id>/edit", methods=["GET", "POST"])
@login_required
def edit_class(class_id):
    conn = get_db()
    cls = conn.execute("SELECT * FROM classes WHERE id = ?", (class_id,)).fetchone()
    if not cls:
        conn.close()
        flash("Class not found.", "danger")
        return redirect(url_for("classes"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        start_time = request.form.get("start_time", "").strip() or None
        end_time = request.form.get("end_time", "").strip() or None
        late_after = request.form.get("late_after_minutes", "10").strip() or "10"
        try:
            conn.execute(
                "UPDATE classes SET name=?, start_time=?, end_time=?, late_after_minutes=? WHERE id=?",
                (name, start_time, end_time, int(late_after), class_id)
            )
            conn.commit()
            flash("Class updated.", "success")
            conn.close()
            return redirect(url_for("classes"))
        except sqlite3.IntegrityError:
            flash("A class with that name already exists.", "danger")

    conn.close()
    return render_template("class_form.html", cls=cls)


@app.route("/classes/<int:class_id>/delete", methods=["POST"])
@login_required
def delete_class(class_id):
    conn = get_db()
    conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    conn.commit()
    conn.close()
    flash("Class deleted (and its students/attendance records).", "info")
    return redirect(url_for("classes"))


# ---------------------------------------------------------------------
# Student management
# ---------------------------------------------------------------------
@app.route("/students")
@login_required
def students():
    conn = get_db()
    class_filter = request.args.get("class_id", "")

    query = """
        SELECT s.*, c.name AS class_name
        FROM students s
        JOIN classes c ON c.id = s.class_id
    """
    params = []
    if class_filter:
        query += " WHERE s.class_id = ?"
        params.append(class_filter)
    query += " ORDER BY c.name, s.roll_no"

    all_students = conn.execute(query, params).fetchall()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()
    conn.close()

    return render_template(
        "students.html",
        students=all_students,
        classes=all_classes,
        selected_class=class_filter
    )


@app.route("/students/add", methods=["GET", "POST"])
@login_required
def add_student():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()

    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()
        name = request.form.get("name", "").strip()
        class_id = request.form.get("class_id")
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()

        if not roll_no or not name or not class_id:
            flash("Roll number, name, and class are required.", "danger")
        else:
            try:
                conn.execute(
                    """INSERT INTO students (roll_no, name, class_id, email, phone)
                       VALUES (?, ?, ?, ?, ?)""",
                    (roll_no, name, class_id, email, phone)
                )
                conn.commit()
                flash(f'Student "{name}" added.', "success")
                conn.close()
                return redirect(url_for("students"))
            except sqlite3.IntegrityError:
                flash("A student with that roll number already exists in this class.", "danger")

    conn.close()
    return render_template("student_form.html", classes=all_classes, student=None)


@app.route("/students/<int:student_id>/edit", methods=["GET", "POST"])
@login_required
def edit_student(student_id):
    conn = get_db()
    student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()

    if not student:
        conn.close()
        flash("Student not found.", "danger")
        return redirect(url_for("students"))

    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()
        name = request.form.get("name", "").strip()
        class_id = request.form.get("class_id")
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        status = request.form.get("status") or "active"
        if status not in ("active", "graduated", "inactive"):
            status = "active"

        # Only stamp status_set_at when the status actually changes, so the
        # Automatic Data Expiry grace period (data_expiry.py) counts from
        # the real moment someone was marked graduated/inactive.
        status_set_at = student["status_set_at"]
        if status != student["status"]:
            status_set_at = datetime.now().isoformat(timespec="seconds") if status != "active" else None

        try:
            conn.execute(
                """UPDATE students SET roll_no=?, name=?, class_id=?, email=?, phone=?,
                   status=?, status_set_at=? WHERE id = ?""",
                (roll_no, name, class_id, email, phone, status, status_set_at, student_id)
            )
            conn.commit()
            flash("Student updated.", "success")
            conn.close()
            return redirect(url_for("students"))
        except sqlite3.IntegrityError:
            flash("A student with that roll number already exists in this class.", "danger")

    conn.close()
    return render_template("student_form.html", classes=all_classes, student=student)


@app.route("/students/<int:student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    conn = get_db()
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    flash("Student deleted.", "info")
    return redirect(url_for("students"))


# ---------------------------------------------------------------------
# Teacher management
# ---------------------------------------------------------------------
@app.route("/teachers")
@login_required
def teachers():
    conn = get_db()
    all_teachers = conn.execute("SELECT * FROM teachers ORDER BY name").fetchall()
    conn.close()
    return render_template("teachers.html", teachers=all_teachers)


@app.route("/teachers/add", methods=["GET", "POST"])
@login_required
def add_teacher():
    if request.method == "POST":
        teacher_code = request.form.get("teacher_code", "").strip()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()

        if not teacher_code or not name:
            flash("Teacher code and name are required.", "danger")
        else:
            conn = get_db()
            try:
                conn.execute(
                    "INSERT INTO teachers (teacher_code, name, email, phone) VALUES (?, ?, ?, ?)",
                    (teacher_code, name, email, phone)
                )
                conn.commit()
                flash(f'Teacher "{name}" added.', "success")
                conn.close()
                return redirect(url_for("teachers"))
            except sqlite3.IntegrityError:
                flash("A teacher with that teacher code already exists.", "danger")
            conn.close()

    return render_template("teacher_form.html", teacher=None)


@app.route("/teachers/<int:teacher_id>/edit", methods=["GET", "POST"])
@login_required
def edit_teacher(teacher_id):
    conn = get_db()
    teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
    if not teacher:
        conn.close()
        flash("Teacher not found.", "danger")
        return redirect(url_for("teachers"))

    if request.method == "POST":
        teacher_code = request.form.get("teacher_code", "").strip()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        try:
            conn.execute(
                "UPDATE teachers SET teacher_code=?, name=?, email=?, phone=? WHERE id=?",
                (teacher_code, name, email, phone, teacher_id)
            )
            conn.commit()
            flash("Teacher updated.", "success")
            conn.close()
            return redirect(url_for("teachers"))
        except sqlite3.IntegrityError:
            flash("A teacher with that teacher code already exists.", "danger")

    conn.close()
    return render_template("teacher_form.html", teacher=teacher)


@app.route("/teachers/<int:teacher_id>/delete", methods=["POST"])
@login_required
def delete_teacher(teacher_id):
    conn = get_db()
    conn.execute("DELETE FROM teachers WHERE id = ?", (teacher_id,))
    conn.commit()
    conn.close()
    flash("Teacher deleted.", "info")
    return redirect(url_for("teachers"))


@app.route("/teachers/face/capture/<int:teacher_id>")
@login_required
def teacher_face_capture(teacher_id):
    conn = get_db()
    teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
    conn.close()
    if not teacher:
        flash("Teacher not found.", "danger")
        return redirect(url_for("teachers"))
    return render_template(
        "face_capture.html",
        person=teacher,
        person_kind="teacher",
        save_url=url_for("teacher_face_capture_save", teacher_id=teacher_id),
        back_url=url_for("teachers"),
        samples_target=SAMPLES_PER_TEACHER,
        existing_count=face_sample_count(teacher_id, TEACHER_DATASET_DIR)
    )


@app.route("/teachers/face/capture/<int:teacher_id>/save", methods=["POST"])
@login_required
def teacher_face_capture_save(teacher_id):
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"success": False, "message": "No image received."}), 400

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Could not read image."}), 400

    # Modern multi-angle detection & quality check (Item 1, 13, 15)
    detections = face_engine.detect_faces(img)
    if len(detections) == 0:
        return jsonify({"success": False, "message": "No face detected — please look directly at the camera."})
    if len(detections) > 1:
        return jsonify({"success": False, "message": "Multiple faces detected — please ensure only you are in frame."})

    det = detections[0]
    x, y, w, h = det["box"]
    quality_res = face_quality.assess_face_quality(img, det["box"], det["landmarks"], detection_conf=det["score"])
    if not quality_res["passed"]:
        reason_msg = quality_res["reasons"][0] if quality_res["reasons"] else "Face quality insufficient — re-verifying"
        return jsonify({"success": False, "message": reason_msg, "quality": quality_res["quality_score"]})

    # Save high-quality sample
    ensure_face_dirs()
    teacher_dir = os.path.join(TEACHER_DATASET_DIR, str(teacher_id))
    os.makedirs(teacher_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], (200, 200))
    next_index = face_sample_count(teacher_id, TEACHER_DATASET_DIR) + 1
    cv2.imwrite(os.path.join(teacher_dir, f"{next_index}.jpg"), face_img)

    # Extract modern 128-D SFace embedding
    _, emb = face_engine.extract_embedding(img, det)
    if emb is not None:
        conn = get_db()
        conn.execute(
            "INSERT INTO face_embeddings (person_type, person_id, pose_tag, quality_score, embedding_blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("teacher", teacher_id, quality_res["pose"], quality_res["quality_score"], emb.tobytes(), datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
        conn.close()

    return jsonify({
        "success": True,
        "count": next_index,
        "pose": quality_res["pose"],
        "quality": quality_res["quality_score"],
        "target": config.SAMPLES_PER_TEACHER
    })


@app.route("/teachers/face/enroll_video/<int:teacher_id>", methods=["POST"])
@login_required
def teacher_face_enroll_video(teacher_id):
    """Process optional 60-second multi-angle video enrollment for teachers (Item 2, 15)."""
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"success": False, "message": "No video received."}), 400

    ensure_face_dirs()
    temp_path = os.path.join(FACE_DIR, "temp", f"teacher_vid_{teacher_id}_{int(time.time())}.webm")
    video_file.save(temp_path)

    extracted = face_engine.extract_diverse_enrollment_frames_from_video(temp_path, target_count=config.SAMPLES_PER_TEACHER)
    teacher_dir = os.path.join(TEACHER_DATASET_DIR, str(teacher_id))
    os.makedirs(teacher_dir, exist_ok=True)

    conn = get_db()
    now_iso = datetime.now().isoformat(timespec="seconds")
    curr_count = face_sample_count(teacher_id, TEACHER_DATASET_DIR)
    saved_count = 0

    for (frame, box, q, emb) in extracted:
        curr_count += 1
        x, y, w, h = box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], (200, 200))
        cv2.imwrite(os.path.join(teacher_dir, f"{curr_count}.jpg"), face_img)
        saved_count += 1

        if emb is not None:
            conn.execute(
                "INSERT INTO face_embeddings (person_type, person_id, pose_tag, quality_score, embedding_blob, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("teacher", teacher_id, q["pose"], q["quality_score"], emb.tobytes(), now_iso)
            )

    conn.commit()
    conn.close()

    try:
        os.remove(temp_path)
    except Exception:
        pass

    face_engine.load_gallery(is_teacher=True, force_reload=True)
    return jsonify({
        "success": True,
        "count": curr_count,
        "extracted": saved_count,
        "message": f"Successfully extracted {saved_count} diverse face samples across angles."
    })


@app.route("/teachers/face/capture/<int:teacher_id>/delete", methods=["POST"])
@login_required
def teacher_face_capture_delete(teacher_id):
    teacher_dir = os.path.join(TEACHER_DATASET_DIR, str(teacher_id))
    if os.path.isdir(teacher_dir):
        shutil.rmtree(teacher_dir)
        flash("Face data cleared for this teacher.", "info")
    return redirect(url_for("teachers"))


@app.route("/teachers/face/train", methods=["POST"])
@login_required
def teacher_face_train():
    ensure_face_dirs()
    faces, labels = [], []

    for entry in os.listdir(TEACHER_DATASET_DIR):
        teacher_dir = os.path.join(TEACHER_DATASET_DIR, entry)
        if not os.path.isdir(teacher_dir):
            continue
        try:
            teacher_id = int(entry)
        except ValueError:
            continue
        for img_path in glob.glob(os.path.join(teacher_dir, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(teacher_id)

    if len(faces) < 2 or len(set(labels)) < 1:
        flash("Not enough teacher face samples yet. Capture some first.", "danger")
        return redirect(url_for("teachers"))

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    os.makedirs(FACE_DIR, exist_ok=True)
    model.write(TEACHER_TRAINER_FILE)
    load_teacher_recognizer()
    face_engine.load_gallery(is_teacher=True, force_reload=True)

    flash(f"Teacher recognition model trained on {len(faces)} images across {len(set(labels))} teacher(s).", "success")
    return redirect(url_for("teachers"))


# ---------------------------------------------------------------------
# Face recognition — enrollment
# ---------------------------------------------------------------------
@app.route("/face")
@login_required
def face_setup():
    conn = get_db()
    all_students = conn.execute("""
        SELECT s.*, c.name AS class_name
        FROM students s
        JOIN classes c ON c.id = s.class_id
        ORDER BY c.name, s.roll_no
    """).fetchall()
    conn.close()

    samples = {s["id"]: face_sample_count(s["id"]) for s in all_students}
    model_trained = os.path.exists(TRAINER_FILE)
    trained_at = None
    if model_trained:
        trained_at = datetime.fromtimestamp(os.path.getmtime(TRAINER_FILE)).strftime("%Y-%m-%d %H:%M")

    return render_template(
        "face_setup.html",
        students=all_students,
        samples=samples,
        samples_target=SAMPLES_PER_STUDENT,
        model_trained=model_trained,
        trained_at=trained_at
    )


@app.route("/face/capture/<int:student_id>")
@login_required
def face_capture(student_id):
    conn = get_db()
    student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("face_setup"))
    return render_template(
        "face_capture.html",
        person=student,
        person_kind="student",
        save_url=url_for("face_capture_save", student_id=student_id),
        back_url=url_for("face_setup"),
        samples_target=SAMPLES_PER_STUDENT,
        existing_count=face_sample_count(student_id)
    )


@app.route("/face/capture/<int:student_id>/save", methods=["POST"])
@login_required
def face_capture_save(student_id):
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"success": False, "message": "No image received."}), 400

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Could not read image."}), 400

    # Modern multi-angle detection & quality check (Item 1, 13)
    detections = face_engine.detect_faces(img)
    if len(detections) == 0:
        return jsonify({"success": False, "message": "No face detected — please look directly at the camera."})
    if len(detections) > 1:
        return jsonify({"success": False, "message": "Multiple faces detected — please ensure only you are in frame."})

    det = detections[0]
    x, y, w, h = det["box"]
    quality_res = face_quality.assess_face_quality(img, det["box"], det["landmarks"], detection_conf=det["score"])
    if not quality_res["passed"]:
        reason_msg = quality_res["reasons"][0] if quality_res["reasons"] else "Face quality insufficient — re-verifying"
        return jsonify({"success": False, "message": reason_msg, "quality": quality_res["quality_score"]})

    # Save high-quality sample
    ensure_face_dirs()
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], (200, 200))
    next_index = face_sample_count(student_id) + 1
    cv2.imwrite(os.path.join(student_dir, f"{next_index}.jpg"), face_img)

    # Extract modern 128-D SFace embedding
    _, emb = face_engine.extract_embedding(img, det)
    conn = get_db()
    now_iso = datetime.now().isoformat(timespec="seconds")
    if emb is not None:
        conn.execute(
            "INSERT INTO face_embeddings (person_type, person_id, pose_tag, quality_score, embedding_blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("student", student_id, quality_res["pose"], quality_res["quality_score"], emb.tobytes(), now_iso)
        )

    # Re-enrolling clears any prior "purged by retention policy" marker.
    conn.execute("UPDATE students SET face_deleted_at = NULL WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "count": next_index,
        "pose": quality_res["pose"],
        "quality": quality_res["quality_score"],
        "target": config.SAMPLES_PER_STUDENT
    })


@app.route("/face/enroll_video/<int:student_id>", methods=["POST"])
@login_required
def face_enroll_video(student_id):
    """Process optional 60-second multi-angle video enrollment for students (Item 2)."""
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"success": False, "message": "No video received."}), 400

    ensure_face_dirs()
    temp_path = os.path.join(FACE_DIR, "temp", f"student_vid_{student_id}_{int(time.time())}.webm")
    video_file.save(temp_path)

    extracted = face_engine.extract_diverse_enrollment_frames_from_video(temp_path, target_count=config.SAMPLES_PER_STUDENT)
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)

    conn = get_db()
    now_iso = datetime.now().isoformat(timespec="seconds")
    curr_count = face_sample_count(student_id)
    saved_count = 0

    for (frame, box, q, emb) in extracted:
        curr_count += 1
        x, y, w, h = box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], (200, 200))
        cv2.imwrite(os.path.join(student_dir, f"{curr_count}.jpg"), face_img)
        saved_count += 1

        if emb is not None:
            conn.execute(
                "INSERT INTO face_embeddings (person_type, person_id, pose_tag, quality_score, embedding_blob, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("student", student_id, q["pose"], q["quality_score"], emb.tobytes(), now_iso)
            )

    conn.execute("UPDATE students SET face_deleted_at = NULL WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()

    try:
        os.remove(temp_path)
    except Exception:
        pass

    face_engine.load_gallery(is_teacher=False, force_reload=True)
    return jsonify({
        "success": True,
        "count": curr_count,
        "extracted": saved_count,
        "message": f"Successfully extracted {saved_count} diverse face samples across angles."
    })


@app.route("/face/capture/<int:student_id>/delete", methods=["POST"])
@login_required
def face_capture_delete(student_id):
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    if os.path.isdir(student_dir):
        shutil.rmtree(student_dir)
        flash("Face data cleared for this student. You can recapture it any time.", "info")
    return redirect(url_for("face_setup"))


@app.route("/face/train", methods=["POST"])
@login_required
def face_train():
    ensure_face_dirs()
    faces, labels = [], []

    for entry in os.listdir(DATASET_DIR):
        student_dir = os.path.join(DATASET_DIR, entry)
        if not os.path.isdir(student_dir):
            continue
        try:
            student_id = int(entry)
        except ValueError:
            continue
        for img_path in glob.glob(os.path.join(student_dir, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(student_id)

    if len(faces) < 2 or len(set(labels)) < 1:
        flash("Not enough face samples yet. Capture at least a dozen images for one or more students first.", "danger")
        return redirect(url_for("face_setup"))

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    os.makedirs(FACE_DIR, exist_ok=True)
    model.write(TRAINER_FILE)
    load_recognizer()
    face_engine.load_gallery(is_teacher=False, force_reload=True)

    flash(f"Model trained on {len(faces)} images across {len(set(labels))} student(s). "
          f"You're ready to use Live Attendance.", "success")
    return redirect(url_for("face_setup"))


# ---------------------------------------------------------------------
# Face recognition — live camera attendance
# ---------------------------------------------------------------------
@app.route("/attendance/live")
@login_required
def live_attendance():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()

    class_id = request.args.get("class_id") or (str(all_classes[0]["id"]) if all_classes else "")
    att_date = request.args.get("att_date") or date.today().isoformat()

    selected_cls = None
    if class_id:
        selected_cls = conn.execute("SELECT * FROM classes WHERE id = ?", (class_id,)).fetchone()
    conn.close()

    # Attendance can only be marked "live" for today. The page itself does
    # its own teacher-first detection via /attendance/smart/state and
    # /attendance/smart/start_session (see live_attendance.html) — the
    # server-side enforcement that a teacher must actually be confirmed
    # present lives in live_recognize() below, regardless of what the
    # client-side UI shows.
    is_today = (att_date == date.today().isoformat()) or (att_date == datetime.now().date().isoformat())

    return render_template(
        "live_attendance.html",
        classes=all_classes,
        selected_class=str(class_id),
        att_date=att_date,
        model_trained=os.path.exists(TRAINER_FILE),
        selected_cls=selected_cls,
        is_today=is_today
    )



@app.route("/attendance/live/roster")
@login_required
def live_roster():
    class_id = request.args.get("class_id")
    att_date = request.args.get("att_date") or date.today().isoformat()
    if not class_id:
        return jsonify([])

    conn = get_db()
    rows = conn.execute("""
        SELECT s.id, s.roll_no, s.name, COALESCE(a.status, 'Not Marked') AS status
        FROM students s
        LEFT JOIN attendance a ON a.student_id = s.id AND a.att_date = ?
        WHERE s.class_id = ?
        ORDER BY s.roll_no
    """, (att_date, class_id)).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


@app.route("/attendance/live/recognize", methods=["POST"])
@login_required
def live_recognize():
    if recognizer is None:
        return jsonify({"success": False, "message": "Face model has not been trained yet."}), 400

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    class_id = data.get("class_id")
    att_date = data.get("att_date") or date.today().isoformat()

    if not image_data or not class_id:
        return jsonify({"success": False, "message": "Missing image or class."}), 400

    # Only apply live "is it past start time?" logic when marking today.
    # Attendance taken for a past/future date has no reliable "now" to compare.
    is_today = (att_date == date.today().isoformat()) or (att_date == datetime.now().date().isoformat())

    conn = get_db()

    # A teacher must be face-confirmed present (via Smart Attendance) for
    # this class's current period before student attendance can be marked
    # here. This only applies to "today" — there's no live period to be
    # present for on a past/future date.
    if is_today:
        active_session = get_active_teacher_session(conn, class_id)
        if not active_session:
            conn.close()
            return jsonify({
                "success": False,
                "state": "WAITING_FOR_TEACHER",
                "message": "Student attendance is locked: teacher has not been confirmed and the 30-minute grace period has not elapsed."
            }), 403

    img = decode_base64_image(image_data)
    if img is None:
        conn.close()
        return jsonify({"success": False, "message": "Could not read image."}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    cls_row = conn.execute("SELECT * FROM classes WHERE id = ?", (class_id,)).fetchone()
    class_students = {
        row["id"]: row for row in
        conn.execute("SELECT * FROM students WHERE class_id = ?", (class_id,)).fetchall()
    }
    already_marked = {
        row["student_id"] for row in
        conn.execute(
            "SELECT student_id FROM attendance WHERE att_date = ? AND class_id = ?",
            (att_date, class_id)
        ).fetchall()
    }

    # Multi-person Tracking and Attendance Protection (Item 4, 5, 6, 7, 16, 21)
    trk = LIVE_ATTENDANCE_TRACKER.setdefault(class_id, tracker.ClassroomTracker())
    student_gallery = face_engine.load_gallery(is_teacher=False)
    tracked_persons = trk.process_frame(img, gallery=student_gallery, roster=class_students)

    results = []
    newly_marked = []

    # Fallback to legacy recognizer when faces are mocked or provided
    if not tracked_persons and len(faces) > 0 and recognizer is not None:
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            try:
                label, confidence = recognizer.predict(face_img)
            except Exception:
                label, confidence = 0, 999
            box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

            if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD and label in class_students:
                student = class_students[label]
                if label in already_marked:
                    status = "already_marked"
                    att_status = None
                else:
                    att_status = determine_live_status(cls_row) if is_today else "Present"
                    conn.execute("""
                        INSERT INTO attendance (student_id, class_id, att_date, status)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(student_id, att_date) DO NOTHING
                    """, (label, class_id, att_date, att_status))
                    already_marked.add(label)
                    status = "marked"
                    newly_marked.append({
                        "id": label, "name": student["name"], "roll_no": student["roll_no"],
                        "att_status": att_status
                    })
                results.append({
                    **box, "track_id": 1, "name": student["name"], "roll_no": student["roll_no"],
                    "status": status, "att_status": att_status, "confidence": round(float(confidence), 1)
                })
            else:
                results.append({
                    **box, "track_id": 1, "name": "Unknown", "roll_no": "",
                    "status": "unknown", "att_status": None, "confidence": round(float(confidence), 1)
                })


    for p in tracked_persons:
        bx, by, bw, bh = p["box"]
        box = {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)}
        crop = gray[max(0, by):by + bh, max(0, bx):bx + bw]
        crop_resized = cv2.resize(crop, (200, 200)) if crop.size > 0 else None

        # Cross-check teacher match
        if crop_resized is not None:
            t_match = better_teacher_match(crop_resized, (1.0 - (p["confidence"] / 100.0)) * 100.0, conn)
            if t_match:
                teacher, t_conf = t_match
                results.append({
                    **box, "track_id": p["track_id"], "name": teacher["name"], "roll_no": "",
                    "status": "teacher", "att_status": None, "confidence": round(float(t_conf), 1)
                })
                continue

        label = p["student_id"]
        # Strict 6-Gate Attendance Protection (Item 16)
        if p["status"] == "VERIFIED" and label in class_students:
            student = class_students[label]
            if label in already_marked:
                status = "already_marked"
                att_status = None
            else:
                att_status = determine_live_status(cls_row) if is_today else "Present"
                conn.execute("""
                    INSERT INTO attendance (student_id, class_id, att_date, status)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(student_id, att_date) DO NOTHING
                """, (label, class_id, att_date, att_status))
                already_marked.add(label)
                status = "marked"
                newly_marked.append({
                    "id": label, "name": student["name"], "roll_no": student["roll_no"],
                    "att_status": att_status
                })
            results.append({
                **box, "track_id": p["track_id"], "name": student["name"], "roll_no": student["roll_no"],
                "status": status, "att_status": att_status,
                "confidence": p["confidence"], "identity_score": p["identity_score"]
            })
        elif p["status"] == "UNDER_REVIEW":
            results.append({
                **box, "track_id": p["track_id"], "name": f"Track #{p['track_id']} (Verifying)",
                "roll_no": "", "status": "review", "att_status": None,
                "confidence": p["confidence"], "identity_score": p["identity_score"]
            })
        else:
            results.append({
                **box, "track_id": p["track_id"], "name": "Unknown",
                "roll_no": "", "status": "unknown", "att_status": None,
                "confidence": p["confidence"], "identity_score": p["identity_score"]
            })

    if newly_marked:
        conn.commit()
    conn.close()

    return jsonify({"success": True, "faces": results, "newly_marked": newly_marked})


# ---------------------------------------------------------------------
# Timetable management
# ---------------------------------------------------------------------
@app.route("/timetable")
@login_required
def timetable():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()
    class_id = request.args.get("class_id") or (str(all_classes[0]["id"]) if all_classes else "")

    periods = []
    all_teachers = conn.execute("SELECT * FROM teachers ORDER BY name").fetchall()
    if class_id:
        periods = conn.execute("""
            SELECT t.*, te.name AS teacher_name
            FROM timetable t
            LEFT JOIN teachers te ON te.id = t.teacher_id
            WHERE t.class_id = ?
            ORDER BY t.start_time
        """, (class_id,)).fetchall()
    conn.close()

    return render_template(
        "timetable.html",
        classes=all_classes,
        teachers=all_teachers,
        periods=periods,
        selected_class=str(class_id)
    )


@app.route("/timetable/generate", methods=["POST"])
@login_required
def timetable_generate():
    class_id = request.form.get("class_id")
    if not class_id:
        flash("Pick a class first.", "danger")
        return redirect(url_for("timetable"))

    def _do_gen():
        with get_db_transaction() as conn:
            return generate_default_timetable(conn, class_id)

    count = execute_with_retry(_do_gen)
    flash(f"Generated a default day with {count} period(s) "
          f"({DEFAULT_DAY_START}–{DEFAULT_DAY_END}, tea break, lunch break). "
          f"Now fill in the subject and teacher for each class period.", "success")
    return redirect(url_for("timetable", class_id=class_id))


@app.route("/timetable/add", methods=["POST"])
@login_required
def timetable_add():
    class_id = request.form.get("class_id")
    period_number = request.form.get("period_number", "1")
    period_type = request.form.get("period_type", "class")
    subject = request.form.get("subject", "").strip() or None
    teacher_id = request.form.get("teacher_id") or None
    start_time = request.form.get("start_time")
    end_time = request.form.get("end_time")

    if not (class_id and start_time and end_time):
        flash("Start time and end time are required.", "danger")
        return redirect(url_for("timetable", class_id=class_id))

    def _do_add():
        with get_db_transaction() as conn:
            conn.execute("""
                INSERT INTO timetable (class_id, period_number, period_type, subject, teacher_id, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (class_id, period_number, period_type, subject, teacher_id, start_time, end_time))

    execute_with_retry(_do_add)
    flash("Period added to timetable.", "success")
    return redirect(url_for("timetable", class_id=class_id))


@app.route("/timetable/<int:period_id>/update", methods=["POST"])
@login_required
def timetable_update(period_id):
    subject = request.form.get("subject", "").strip() or None
    teacher_id = request.form.get("teacher_id") or None
    class_id = request.form.get("class_id")

    def _do_update():
        with get_db_transaction() as conn:
            conn.execute(
                "UPDATE timetable SET subject = ?, teacher_id = ? WHERE id = ?",
                (subject, teacher_id, period_id)
            )

    execute_with_retry(_do_update)
    flash("Period updated.", "success")
    return redirect(url_for("timetable", class_id=class_id))


@app.route("/timetable/<int:period_id>/delete", methods=["POST"])
@login_required
def timetable_delete(period_id):
    class_id = request.form.get("class_id")

    def _do_delete():
        with get_db_transaction() as conn:
            conn.execute("DELETE FROM timetable WHERE id = ?", (period_id,))

    execute_with_retry(_do_delete)
    flash("Period removed.", "info")
    return redirect(url_for("timetable", class_id=class_id))


# ---------------------------------------------------------------------
# Smart period attendance — camera detects the teacher first (to confirm
# which subject/period is live), then switches to recognizing students
# ---------------------------------------------------------------------
@app.route("/attendance/smart")
@login_required
def smart_attendance():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()
    class_id = request.args.get("class_id") or (str(all_classes[0]["id"]) if all_classes else "")
    for cls in all_classes:
        ensure_auto_attendance(conn, cls["id"])
    teacher_sessions = list_active_teacher_sessions(conn)
    conn.close()

    return render_template(
        "smart_attendance.html",
        classes=all_classes,
        selected_class=str(class_id),
        model_trained=os.path.exists(TRAINER_FILE),
        teacher_model_trained=os.path.exists(TEACHER_TRAINER_FILE),
        teacher_sessions=teacher_sessions,
        auto_absent_grace_minutes=AUTO_ABSENT_GRACE_MINUTES
    )


@app.route("/attendance/smart/state")
@login_required
def smart_attendance_state():
    class_id = request.args.get("class_id")
    if not class_id:
        return jsonify({
            "state": "IDLE",
            "period": None,
            "session": None,
            "teacher_status": "none",
            "period_start": None,
            "period_end": None,
            "grace_deadline": None,
            "remaining_seconds": 0,
            "student_attendance_allowed": False,
            "message": "No class selected."
        })

    conn = get_db()
    now = datetime.now()
    today = now.date().isoformat()
    period = get_current_period(conn, class_id, now=now)

    if not period:
        conn.close()
        return jsonify({
            "state": "IDLE",
            "period": None,
            "session": None,
            "teacher_status": "none",
            "period_start": None,
            "period_end": None,
            "grace_deadline": None,
            "remaining_seconds": 0,
            "student_attendance_allowed": False,
            "message": "No period scheduled right now."
        })

    if period["period_type"] == "break":
        conn.close()
        period_dict = dict(period) if hasattr(period, "keys") else (period or {})
        return jsonify({
            "state": "IDLE",
            "period": period,
            "session": None,
            "teacher_status": "none",
            "period_start": period["start_time"],
            "period_end": period["end_time"],
            "grace_deadline": None,
            "remaining_seconds": 0,
            "student_attendance_allowed": False,
            "message": f"{period_dict.get('subject') or 'Break'} ({period['start_time']}–{period['end_time']}) — no attendance during breaks."
        })

    timing = get_period_timing(period, now=now)

    # Check/trigger auto-attendance if 30 minutes have elapsed without a teacher
    ensure_auto_attendance(conn, class_id, now=now)

    session_row = conn.execute("""
        SELECT ps.*, t.name AS teacher_name
        FROM period_sessions ps
        LEFT JOIN teachers t ON t.id = ps.teacher_id
        WHERE ps.timetable_id = ? AND ps.session_date = ?
    """, (period["id"], today)).fetchone()

    session_data = None
    if session_row:
        present_count = conn.execute(
            "SELECT COUNT(*) c FROM period_attendance WHERE session_id = ?",
            (session_row["id"],)
        ).fetchone()["c"]
        total_students = conn.execute(
            "SELECT COUNT(*) c FROM students WHERE class_id = ?", (class_id,)
        ).fetchone()["c"]
        session_data = {
            "id": session_row["id"],
            "teacher_name": session_row["teacher_name"],
            "teacher_status": session_row["teacher_status"],
            "auto_marked": bool(session_row["auto_marked"]),
            "started_at": session_row["started_at"],
            "ended_at": session_row["ended_at"],
            "present_count": present_count,
            "total_students": total_students
        }

    # Determine authoritative state
    if session_row and session_row["ended_at"]:
        state = "SESSION_ENDED"
        teacher_status = session_row["teacher_status"]
        student_attendance_allowed = False
        remaining_seconds = 0
        message = "This period has ended — attendance is closed."
    elif session_row and session_row["teacher_id"] and session_row["teacher_status"] == "late":
        state = "TEACHER_LATE"
        teacher_status = "late"
        student_attendance_allowed = True
        remaining_seconds = 0
        message = f"Teacher {session_row['teacher_name']} arrived late. Student attendance active."
    elif session_row and session_row["teacher_id"]:
        state = "TEACHER_CONFIRMED"
        teacher_status = session_row["teacher_status"] or "present"
        student_attendance_allowed = True
        remaining_seconds = 0
        message = f"Teacher {session_row['teacher_name']} verified. Student attendance enabled."
    elif session_row and session_row["teacher_status"] == "absent":
        state = "TEACHER_ABSENT_AUTO_ATTENDANCE"
        teacher_status = "absent"
        student_attendance_allowed = True
        remaining_seconds = 0
        message = "Teacher absent (30-minute grace elapsed). Automatic student attendance active."
    elif timing and not timing["is_past_grace"]:
        state = "WAITING_FOR_TEACHER"
        teacher_status = "none"
        student_attendance_allowed = False
        remaining_seconds = timing["remaining_seconds"]
        mins = remaining_seconds // 60
        secs = remaining_seconds % 60
        message = f"Waiting for teacher. Grace period remaining: {mins:02d}:{secs:02d}."
    else:
        state = "TEACHER_ABSENT_AUTO_ATTENDANCE"
        teacher_status = "absent"
        student_attendance_allowed = True
        remaining_seconds = 0
        message = "Teacher absent (30-minute grace elapsed). Automatic student attendance active."

    conn.close()

    return jsonify({
        "state": state,
        "teacher_status": teacher_status,
        "period": period,
        "session": session_data,
        "period_start": timing["period_start_str"] if timing else period["start_time"],
        "period_end": timing["period_end_str"] if timing else period["end_time"],
        "grace_deadline": timing["grace_deadline_str"] if timing else None,
        "remaining_seconds": remaining_seconds,
        "student_attendance_allowed": student_attendance_allowed,
        "message": message
    })


@app.route("/attendance/smart/start_session", methods=["POST"])
@login_required
def smart_start_session():
    if teacher_recognizer is None:
        return jsonify({"success": False, "message": "Teacher recognition model has not been trained yet."}), 400

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    class_id = data.get("class_id")

    if not image_data or not class_id:
        return jsonify({"success": False, "message": "Missing image or class."}), 400

    now = datetime.now()
    conn = get_db()
    period = get_current_period(conn, class_id, now=now)
    if not period or period["period_type"] != "class":
        conn.close()
        return jsonify({"success": False, "message": "No class period is scheduled right now."})

    today = now.date().isoformat()
    timing = get_period_timing(period, now=now)

    existing = conn.execute("""
        SELECT ps.*, t.name AS teacher_name FROM period_sessions ps
        LEFT JOIN teachers t ON t.id = ps.teacher_id
        WHERE ps.timetable_id = ? AND ps.session_date = ?
    """, (period["id"], today)).fetchone()

    # If teacher already confirmed on this session
    if existing and existing["teacher_id"]:
        conn.close()
        return jsonify({
            "success": True, "already_active": True,
            "session_id": existing["id"], "teacher_name": existing["teacher_name"],
            "teacher_status": existing["teacher_status"],
            "subject": period["subject"]
        })

    img = decode_base64_image(image_data)
    if img is None:
        conn.close()
        return jsonify({"success": False, "message": "Could not read image."}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        conn.close()
        return jsonify({"success": False, "message": "No face detected. Ask the teacher to look at the camera."})

    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
    label, confidence = teacher_recognizer.predict(face_img)

    if confidence > RECOGNITION_CONFIDENCE_THRESHOLD:
        conn.close()
        return jsonify({"success": False, "message": "Teacher not recognized. Try again in better lighting."})

    teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (label,)).fetchone()
    if not teacher:
        conn.close()
        return jsonify({"success": False, "message": "Recognized face does not match any enrolled teacher."})

    # Strict timetable authorization match:
    if not period["teacher_id"]:
        conn.close()
        return jsonify({
            "success": False,
            "message": "No teacher is assigned to this period in the timetable. Student attendance cannot be taken."
        })

    if int(period["teacher_id"]) != teacher["id"]:
        conn.close()
        return jsonify({
            "success": False,
            "message": f'{teacher["name"]} is not the teacher scheduled for this period'
                       f'{" (" + period["subject"] + ")" if (dict(period).get("subject") if hasattr(period, "keys") else (period or {}).get("subject")) else ""}. '
                       f'Student attendance cannot be taken.'
        })

    now_iso = now.isoformat(timespec="seconds")
    is_late_arrival = bool(timing and timing["is_past_grace"])

    if is_late_arrival:
        # Teacher arrived AFTER 30 minutes grace deadline:
        # Follow late-teacher policy: mark teacher_status = 'late', do not reset grace timer
        if existing:
            conn.execute(
                "UPDATE period_sessions SET teacher_id = ?, teacher_status = 'late' WHERE id = ?",
                (teacher["id"], existing["id"])
            )
            session_id = existing["id"]
        else:
            started_iso = timing["period_start_dt"].isoformat(timespec="seconds") if timing else now_iso
            conn.execute("""
                INSERT INTO period_sessions (class_id, timetable_id, session_date, teacher_id, started_at, teacher_status, auto_marked)
                VALUES (?, ?, ?, ?, ?, 'late', 1)
            """, (class_id, period["id"], today, teacher["id"], started_iso))
            session_id = conn.execute(
                "SELECT id FROM period_sessions WHERE timetable_id = ? AND session_date = ?",
                (period["id"], today)
            ).fetchone()["id"]

        conn.commit()
        conn.close()
        return jsonify({
            "success": True, "already_active": False,
            "state": "TEACHER_LATE",
            "session_id": session_id, "teacher_name": teacher["name"],
            "teacher_status": "late",
            "subject": period["subject"], "confidence": round(float(confidence), 1),
            "note": "Teacher arrived late after the 30-minute grace period. Marked as Late."
        })

    # Teacher arrived on time (before 30 minutes)
    if existing:
        conn.execute(
            "UPDATE period_sessions SET teacher_id = ?, teacher_status = 'present' WHERE id = ?",
            (teacher["id"], existing["id"])
        )
        session_id = existing["id"]
    else:
        conn.execute("""
            INSERT INTO period_sessions (class_id, timetable_id, session_date, teacher_id, started_at, teacher_status)
            VALUES (?, ?, ?, ?, ?, 'present')
        """, (class_id, period["id"], today, teacher["id"], now_iso))
        session_id = conn.execute(
            "SELECT id FROM period_sessions WHERE timetable_id = ? AND session_date = ?",
            (period["id"], today)
        ).fetchone()["id"]

    conn.commit()
    conn.close()

    return jsonify({
        "success": True, "already_active": False,
        "state": "TEACHER_CONFIRMED",
        "session_id": session_id, "teacher_name": teacher["name"],
        "teacher_status": "present",
        "subject": period["subject"],
        "confidence": round(float(confidence), 1)
    })


@app.route("/attendance/smart/mark_student", methods=["POST"])
@login_required
def smart_mark_student():
    if recognizer is None:
        return jsonify({"success": False, "message": "Student recognition model has not been trained yet."}), 400

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    session_id = data.get("session_id")
    if not image_data or not session_id:
        return jsonify({"success": False, "message": "Missing image or session."}), 400

    conn = get_db()
    sess = conn.execute("SELECT * FROM period_sessions WHERE id = ?", (session_id,)).fetchone()
    if not sess:
        conn.close()
        return jsonify({"success": False, "message": "Session not found."}), 404

    if sess["ended_at"] is not None:
        conn.close()
        return jsonify({
            "success": False,
            "state": "SESSION_ENDED",
            "message": "This period session has ended. Student attendance cannot be marked."
        }), 400

    # Retrieve timetable row to evaluate grace deadline
    tt = conn.execute("SELECT * FROM timetable WHERE id = ?", (sess["timetable_id"],)).fetchone()
    now = datetime.now()
    timing = get_period_timing(tt, now=now) if tt else None

    # Server-side enforcement:
    # Student attendance is ONLY allowed if:
    # 1. Teacher was confirmed present or late (sess["teacher_id"] is not None), OR
    # 2. 30-minute grace period has elapsed without a teacher and auto-attendance policy
    #    is active (sess["teacher_status"] == 'absent' and timing["is_past_grace"])
    allowed = False
    if sess["teacher_id"] is not None:
        allowed = True
    elif sess["teacher_status"] == "absent" and timing and timing["is_past_grace"]:
        allowed = True

    if not allowed:
        conn.close()
        return jsonify({
            "success": False,
            "state": "WAITING_FOR_TEACHER",
            "message": "Student attendance is locked: teacher has not been confirmed and the 30-minute grace period has not elapsed."
        }), 403

    img = decode_base64_image(image_data)
    if img is None:
        conn.close()
        return jsonify({"success": False, "message": "Could not read image."}), 400

    class_students = {
        row["id"]: row for row in
        conn.execute("SELECT * FROM students WHERE class_id = ?", (sess["class_id"],)).fetchall()
    }
    already_marked = {
        row["student_id"] for row in
        conn.execute("SELECT student_id FROM period_attendance WHERE session_id = ?", (session_id,)).fetchall()
    }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    trk = SMART_ATTENDANCE_TRACKER.setdefault(session_id, tracker.ClassroomTracker())
    student_gallery = face_engine.load_gallery(is_teacher=False)
    tracked_persons = trk.process_frame(img, gallery=student_gallery, roster=class_students)

    results = []
    newly_marked = []
    now_iso = datetime.now().isoformat(timespec="seconds")

    # Fallback to legacy recognizer when faces are mocked or provided
    if not tracked_persons and len(faces) > 0 and recognizer is not None:
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            try:
                label, confidence = recognizer.predict(face_img)
            except Exception:
                label, confidence = 0, 999
            box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

            t_match = better_teacher_match(face_img, confidence, conn)
            if t_match:
                teacher, t_conf = t_match
                results.append({
                    **box, "track_id": 1, "name": teacher["name"], "roll_no": "",
                    "status": "teacher", "confidence": round(float(t_conf), 1)
                })
                continue

            if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD and label in class_students:
                student = class_students[label]
                if label in already_marked:
                    status = "already_marked"
                else:
                    conn.execute("""
                        INSERT INTO period_attendance (session_id, student_id, status, marked_at)
                        VALUES (?, ?, 'Present', ?)
                        ON CONFLICT(session_id, student_id) DO NOTHING
                    """, (session_id, label, now_iso))
                    already_marked.add(label)
                    status = "marked"
                    newly_marked.append({"id": label, "name": student["name"], "roll_no": student["roll_no"]})
                results.append({
                    **box, "track_id": 1, "name": student["name"], "roll_no": student["roll_no"],
                    "status": status, "confidence": round(float(confidence), 1)
                })
            else:
                results.append({
                    **box, "track_id": 1, "name": "Unknown", "roll_no": "",
                    "status": "unknown", "confidence": round(float(confidence), 1)
                })

    for p in tracked_persons:
        bx, by, bw, bh = p["box"]
        box = {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)}
        crop = gray[max(0, by):by + bh, max(0, bx):bx + bw]
        crop_resized = cv2.resize(crop, (200, 200)) if crop.size > 0 else None

        if crop_resized is not None:
            t_match = better_teacher_match(crop_resized, (1.0 - (p["confidence"] / 100.0)) * 100.0, conn)
            if t_match:
                teacher, t_conf = t_match
                results.append({
                    **box, "track_id": p["track_id"], "name": teacher["name"], "roll_no": "",
                    "status": "teacher", "confidence": round(float(t_conf), 1)
                })
                continue

        label = p["student_id"]
        if p["status"] == "VERIFIED" and label in class_students:
            student = class_students[label]
            if label in already_marked:
                status = "already_marked"
            else:
                conn.execute("""
                    INSERT INTO period_attendance (session_id, student_id, status, marked_at)
                    VALUES (?, ?, 'Present', ?)
                    ON CONFLICT(session_id, student_id) DO NOTHING
                """, (session_id, label, now_iso))
                already_marked.add(label)
                status = "marked"
                newly_marked.append({"id": label, "name": student["name"], "roll_no": student["roll_no"]})
            results.append({
                **box, "track_id": p["track_id"], "name": student["name"], "roll_no": student["roll_no"],
                "status": status, "confidence": p["confidence"]
            })
        elif p["status"] == "UNDER_REVIEW":
            results.append({
                **box, "track_id": p["track_id"], "name": f"Track #{p['track_id']} (Verifying)",
                "roll_no": "", "status": "review", "confidence": p["confidence"]
            })
        else:
            results.append({
                **box, "track_id": p["track_id"], "name": "Unknown",
                "roll_no": "", "status": "unknown", "confidence": p["confidence"]
            })

    if newly_marked:
        conn.commit()

    present_count = conn.execute(
        "SELECT COUNT(*) c FROM period_attendance WHERE session_id = ?", (session_id,)
    ).fetchone()["c"]
    total_students = len(class_students)
    conn.close()

    return jsonify({
        "success": True, "faces": results, "newly_marked": newly_marked,
        "present_count": present_count, "total_students": total_students
    })



@app.route("/attendance/smart/roster")
@login_required
def smart_roster():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify([])

    conn = get_db()
    sess = conn.execute("SELECT * FROM period_sessions WHERE id = ?", (session_id,)).fetchone()
    if not sess:
        conn.close()
        return jsonify([])

    rows = conn.execute("""
        SELECT s.id, s.roll_no, s.name, COALESCE(pa.status, 'Not Marked') AS status
        FROM students s
        LEFT JOIN period_attendance pa ON pa.student_id = s.id AND pa.session_id = ?
        WHERE s.class_id = ?
        ORDER BY s.roll_no
    """, (session_id, sess["class_id"])).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


@app.route("/attendance/smart/end_session", methods=["POST"])
@login_required
def smart_end_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"success": False, "message": "Missing session id."}), 400

    conn = get_db()
    conn.execute(
        "UPDATE period_sessions SET ended_at = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="seconds"), session_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/attendance/smart/reset_session", methods=["POST"])
@login_required
def smart_reset_session():
    """
    Testing/admin helper: completely delete today's period_sessions row (and
    any period_attendance marked against it) for the given class, so the
    Smart Attendance page goes back to 'awaiting_teacher' for that period
    instead of resuming the session created by an earlier test run today.

    This does not affect other days' records — period_sessions is keyed by
    (timetable_id, session_date), so this only ever removes *today's* row.
    """
    data = request.get_json(silent=True) or {}
    class_id = data.get("class_id")
    if not class_id:
        return jsonify({"success": False, "message": "Missing class."}), 400

    conn = get_db()
    period = get_current_period(conn, class_id)
    if not period or period["period_type"] != "class":
        conn.close()
        return jsonify({"success": False, "message": "No class period is scheduled right now."})

    today = date.today().isoformat()
    session_row = conn.execute(
        "SELECT id FROM period_sessions WHERE timetable_id = ? AND session_date = ?",
        (period["id"], today)
    ).fetchone()

    if not session_row:
        conn.close()
        return jsonify({"success": True, "message": "No session to reset for this period today."})

    conn.execute("DELETE FROM period_attendance WHERE session_id = ?", (session_row["id"],))
    conn.execute("DELETE FROM period_sessions WHERE id = ?", (session_row["id"],))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Session reset — camera will wait for the teacher again."})


# ---------------------------------------------------------------------
# Session history (period-based attendance records)
# ---------------------------------------------------------------------
@app.route("/sessions")
@login_required
def sessions_list():
    conn = get_db()
    rows = conn.execute("""
        SELECT ps.id, ps.session_date, ps.started_at, ps.ended_at,
               ps.teacher_status, ps.auto_marked,
               c.name AS class_name, t.subject, te.name AS teacher_name,
               t.start_time, t.end_time,
               (SELECT COUNT(*) FROM period_attendance pa WHERE pa.session_id = ps.id) AS present_count,
               (SELECT COUNT(*) FROM students s WHERE s.class_id = ps.class_id) AS total_students
        FROM period_sessions ps
        JOIN classes c ON c.id = ps.class_id
        JOIN timetable t ON t.id = ps.timetable_id
        LEFT JOIN teachers te ON te.id = ps.teacher_id
        ORDER BY ps.session_date DESC, t.start_time DESC
        LIMIT 200
    """).fetchall()
    conn.close()
    return render_template("sessions.html", sessions=rows)


@app.route("/sessions/<int:session_id>")
@login_required
def session_detail(session_id):
    conn = get_db()
    sess = conn.execute("""
        SELECT ps.*, c.name AS class_name, t.subject, t.start_time, t.end_time, te.name AS teacher_name
        FROM period_sessions ps
        JOIN classes c ON c.id = ps.class_id
        JOIN timetable t ON t.id = ps.timetable_id
        LEFT JOIN teachers te ON te.id = ps.teacher_id
        WHERE ps.id = ?
    """, (session_id,)).fetchone()

    if not sess:
        conn.close()
        flash("Session not found.", "danger")
        return redirect(url_for("sessions_list"))

    roster = conn.execute("""
        SELECT s.roll_no, s.name, COALESCE(pa.status, 'Absent') AS status, pa.marked_at
        FROM students s
        LEFT JOIN period_attendance pa ON pa.student_id = s.id AND pa.session_id = ?
        WHERE s.class_id = ?
        ORDER BY s.roll_no
    """, (session_id, sess["class_id"])).fetchall()
    conn.close()

    return render_template("session_detail.html", sess=sess, roster=roster)


# ---------------------------------------------------------------------
# Exam Hall Malpractice Detection (USN-attributed, with photo evidence)
# ---------------------------------------------------------------------
def _build_exam_roster(conn, session_id):
    """{student_id: dict(student_id, name, roll_no, seat_no)} of every
    student allocated to this exam session -- the closed set an identified
    face is checked against."""
    rows = conn.execute("""
        SELECT s.id AS student_id, s.name AS name, s.roll_no AS roll_no, esa.seat_no AS seat_no
        FROM exam_seat_allocations esa
        JOIN students s ON s.id = esa.student_id
        WHERE esa.exam_session_id = ?
    """, (session_id,)).fetchall()
    return {r["student_id"]: dict(r) for r in rows}


def _make_exam_identify_fn(roster):
    """Build the identify_fn callback exam_ai calls per detected face.
    Reuses the already-trained student LBPH `recognizer`, but only ever
    returns a match if it's one of the students allocated to *this* room
    -- the same USN can't be "recognized" as a different exam's seat."""
    def identify_fn(gray_face_crop):
        if recognizer is None:
            return None
        try:
            label, confidence = recognizer.predict(gray_face_crop)
        except cv2.error:
            return None
        if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD and label in roster:
            row = roster[label]
            return (label, row["name"], row["roll_no"], row["seat_no"], confidence)
        return None
    return identify_fn


def _make_exam_evidence_fn(session_id):
    """Build the save_evidence_fn callback exam_ai calls to persist a
    cropped photo of the flagged face as proof, alongside the alert."""
    out_dir = os.path.join(EXAM_PHOTOS_DIR, str(session_id))
    os.makedirs(out_dir, exist_ok=True)

    def save_evidence_fn(img_bgr, box, alert_type, identity):
        x, y, bw, bh = box
        pad_x, pad_y = int(bw * 0.4), int(bh * 0.4)
        H, W = img_bgr.shape[:2]
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(W, x + bw + pad_x), min(H, y + bh + pad_y)
        crop = img_bgr[y0:y1, x0:x1]
        if crop is None or crop.size == 0:
            crop = img_bgr
        tag = f"usn-{identity[2]}" if (identity and identity[2]) else "unidentified"
        tag = "".join(c for c in tag if c.isalnum() or c in "-_") or "unidentified"
        fname = f"{int(time.time() * 1000)}_{alert_type}_{tag}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, crop)
        return f"exam_photos/{session_id}/{fname}"
    return save_evidence_fn


@app.route("/exam/rooms", methods=["GET", "POST"])
@login_required
def exam_rooms():
    conn = get_db()
    if request.method == "POST":
        room_name = request.form.get("room_name", "").strip()
        capacity = request.form.get("capacity", "30").strip() or "30"
        location = request.form.get("location", "").strip() or None
        if room_name:
            try:
                conn.execute(
                    "INSERT INTO exam_rooms (room_name, capacity, location) VALUES (?, ?, ?)",
                    (room_name, int(capacity), location)
                )
                conn.commit()
                flash(f'Exam room "{room_name}" added.', "success")
            except sqlite3.IntegrityError:
                flash("That room already exists.", "danger")
        return redirect(url_for("exam_rooms"))

    rooms = conn.execute("""
        SELECT r.id, r.room_name, r.capacity, r.location,
               (SELECT COUNT(*) FROM exam_sessions es WHERE es.room_id = r.id) AS session_count
        FROM exam_rooms r
        ORDER BY r.room_name
    """).fetchall()
    conn.close()
    return render_template("exam_rooms.html", rooms=rooms)


@app.route("/exam/rooms/<int:room_id>/edit", methods=["GET", "POST"])
@login_required
def exam_room_edit(room_id):
    conn = get_db()
    room = conn.execute("SELECT * FROM exam_rooms WHERE id = ?", (room_id,)).fetchone()
    if not room:
        conn.close()
        flash("Room not found.", "danger")
        return redirect(url_for("exam_rooms"))

    if request.method == "POST":
        room_name = request.form.get("room_name", "").strip()
        capacity = request.form.get("capacity", "30").strip() or "30"
        location = request.form.get("location", "").strip() or None
        try:
            conn.execute(
                "UPDATE exam_rooms SET room_name=?, capacity=?, location=? WHERE id=?",
                (room_name, int(capacity), location, room_id)
            )
            conn.commit()
            flash("Room updated.", "success")
            conn.close()
            return redirect(url_for("exam_rooms"))
        except sqlite3.IntegrityError:
            flash("A room with that name already exists.", "danger")

    conn.close()
    return render_template("exam_room_form.html", room=room)


@app.route("/exam/rooms/<int:room_id>/delete", methods=["POST"])
@login_required
def exam_room_delete(room_id):
    conn = get_db()
    conn.execute("DELETE FROM exam_rooms WHERE id = ?", (room_id,))
    conn.commit()
    conn.close()
    flash("Room deleted (and its exam sessions).", "info")
    return redirect(url_for("exam_rooms"))


@app.route("/exam/sessions")
@login_required
def exam_sessions_list():
    conn = get_db()
    rows = conn.execute("""
        SELECT es.id, es.exam_name, es.started_at, es.ended_at, es.created_at,
               r.room_name, c.name AS class_name,
               (SELECT COUNT(*) FROM exam_seat_allocations a WHERE a.exam_session_id = es.id) AS seat_count,
               (SELECT COUNT(*) FROM exam_malpractice_alerts al WHERE al.exam_session_id = es.id) AS alert_count,
               (SELECT COUNT(*) FROM exam_malpractice_alerts al WHERE al.exam_session_id = es.id AND al.severity = 'high') AS high_count
        FROM exam_sessions es
        JOIN exam_rooms r ON r.id = es.room_id
        LEFT JOIN classes c ON c.id = es.class_id
        ORDER BY es.created_at DESC
        LIMIT 200
    """).fetchall()
    conn.close()
    return render_template("exam_sessions.html", sessions=rows)


@app.route("/exam/sessions/new", methods=["GET", "POST"])
@login_required
def exam_session_new():
    conn = get_db()
    rooms = conn.execute("SELECT * FROM exam_rooms ORDER BY room_name").fetchall()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()

    if request.method == "POST":
        exam_name = request.form.get("exam_name", "").strip()
        room_id = request.form.get("room_id")
        class_id = request.form.get("class_id") or None
        if not exam_name or not room_id:
            flash("Exam name and room are required.", "danger")
            conn.close()
            return render_template("exam_session_form.html", rooms=rooms, classes=all_classes)

        cur = conn.execute(
            "INSERT INTO exam_sessions (exam_name, room_id, class_id, created_at) VALUES (?, ?, ?, ?)",
            (exam_name, room_id, class_id, datetime.now().isoformat(timespec="seconds"))
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()
        flash(f'Exam "{exam_name}" created — now allocate seats by USN.', "success")
        return redirect(url_for("exam_session_allocate", session_id=session_id))

    conn.close()
    if not rooms:
        flash("Add an exam room first.", "warning")
        return redirect(url_for("exam_rooms"))
    return render_template("exam_session_form.html", rooms=rooms, classes=all_classes)


@app.route("/exam/sessions/<int:session_id>/allocate", methods=["GET", "POST"])
@login_required
def exam_session_allocate(session_id):
    conn = get_db()
    sess = conn.execute("""
        SELECT es.*, r.room_name, r.capacity
        FROM exam_sessions es JOIN exam_rooms r ON r.id = es.room_id
        WHERE es.id = ?
    """, (session_id,)).fetchone()
    if not sess:
        conn.close()
        flash("Exam session not found.", "danger")
        return redirect(url_for("exam_sessions_list"))

    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()
        seat_no = request.form.get("seat_no", "").strip() or None
        student = None
        if roll_no:
            student = conn.execute(
                "SELECT * FROM students WHERE roll_no = ? COLLATE NOCASE", (roll_no,)
            ).fetchone()
        if not student:
            flash(f'No student found with USN/roll no "{roll_no}".', "danger")
        else:
            def _do_alloc():
                with get_db_transaction() as tx_conn:
                    tx_conn.execute(
                        "INSERT INTO exam_seat_allocations (exam_session_id, student_id, seat_no) VALUES (?, ?, ?)",
                        (session_id, student["id"], seat_no)
                    )
            try:
                execute_with_retry(_do_alloc)
                flash(f'{student["name"]} ({student["roll_no"]}) allocated to this room.', "success")
            except sqlite3.IntegrityError:
                flash(f'{student["name"]} ({student["roll_no"]}) is already allocated to this session.', "warning")
        conn.close()
        return redirect(url_for("exam_session_allocate", session_id=session_id))

    allocations = conn.execute("""
        SELECT a.id AS alloc_id, a.seat_no, s.id AS student_id, s.name, s.roll_no
        FROM exam_seat_allocations a
        JOIN students s ON s.id = a.student_id
        WHERE a.exam_session_id = ?
        ORDER BY (a.seat_no IS NULL), a.seat_no, s.roll_no
    """, (session_id,)).fetchall()

    class_students = []
    if sess["class_id"]:
        allocated_ids = {a["student_id"] for a in allocations}
        class_students = [
            s for s in conn.execute(
                "SELECT * FROM students WHERE class_id = ? ORDER BY roll_no", (sess["class_id"],)
            ).fetchall() if s["id"] not in allocated_ids
        ]
    conn.close()

    return render_template(
        "exam_allocate.html", sess=sess, allocations=allocations, class_students=class_students
    )


@app.route("/exam/sessions/<int:session_id>/allocate/bulk", methods=["POST"])
@login_required
def exam_session_allocate_bulk(session_id):
    """Allocate every remaining (not-yet-allocated) student of the exam's
    linked class in one go, auto-numbering seats 1..N."""
    conn = get_db()
    sess = conn.execute("SELECT * FROM exam_sessions WHERE id = ?", (session_id,)).fetchone()
    if not sess or not sess["class_id"]:
        conn.close()
        flash("This exam session has no linked class to bulk-allocate from.", "danger")
        return redirect(url_for("exam_session_allocate", session_id=session_id))

    existing = conn.execute(
        "SELECT student_id, seat_no FROM exam_seat_allocations WHERE exam_session_id = ?", (session_id,)
    ).fetchall()
    students = conn.execute(
        "SELECT * FROM students WHERE class_id = ? ORDER BY roll_no", (sess["class_id"],)
    ).fetchall()
    conn.close()

    allocated_ids = {r["student_id"] for r in existing}
    next_seat = 1
    used_seat_numbers = {int(r["seat_no"]) for r in existing if r["seat_no"] and r["seat_no"].isdigit()}
    while next_seat in used_seat_numbers:
        next_seat += 1

    to_insert = []
    for s in students:
        if s["id"] in allocated_ids:
            continue
        while next_seat in used_seat_numbers:
            next_seat += 1
        to_insert.append((session_id, s["id"], str(next_seat)))
        used_seat_numbers.add(next_seat)

    def _do_bulk_alloc():
        with get_db_transaction() as tx_conn:
            for item in to_insert:
                tx_conn.execute(
                    "INSERT INTO exam_seat_allocations (exam_session_id, student_id, seat_no) VALUES (?, ?, ?)",
                    item
                )

    execute_with_retry(_do_bulk_alloc)
    flash(f"Allocated {len(to_insert)} student(s) with auto-numbered seats.", "success")
    return redirect(url_for("exam_session_allocate", session_id=session_id))


@app.route("/exam/sessions/<int:session_id>/allocate/<int:alloc_id>/remove", methods=["POST"])
@login_required
def exam_session_allocate_remove(session_id, alloc_id):
    def _do_remove():
        with get_db_transaction() as tx_conn:
            tx_conn.execute(
                "DELETE FROM exam_seat_allocations WHERE id = ? AND exam_session_id = ?",
                (alloc_id, session_id)
            )

    execute_with_retry(_do_remove)
    return redirect(url_for("exam_session_allocate", session_id=session_id))


@app.route("/exam/sessions/<int:session_id>/monitor")
@login_required
def exam_session_monitor(session_id):
    conn = get_db()
    sess = conn.execute("""
        SELECT es.*, r.room_name FROM exam_sessions es
        JOIN exam_rooms r ON r.id = es.room_id WHERE es.id = ?
    """, (session_id,)).fetchone()
    if not sess:
        conn.close()
        flash("Exam session not found.", "danger")
        return redirect(url_for("exam_sessions_list"))
    seat_count = conn.execute(
        "SELECT COUNT(*) AS c FROM exam_seat_allocations WHERE exam_session_id = ?", (session_id,)
    ).fetchone()["c"]
    conn.close()

    if seat_count == 0:
        flash("Allocate at least one student's USN to this room before monitoring.", "warning")
        return redirect(url_for("exam_session_allocate", session_id=session_id))

    return render_template(
        "exam_monitor.html", sess=sess, seat_count=seat_count,
        model_status=exam_ai.get_model_status(),
    )


@app.route("/exam/monitor/start", methods=["POST"])
@login_required
def exam_monitor_start():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"success": False, "message": "Missing session."}), 400
    session_id = int(session_id)

    conn = get_db()
    sess = conn.execute("SELECT * FROM exam_sessions WHERE id = ?", (session_id,)).fetchone()
    if not sess:
        conn.close()
        return jsonify({"success": False, "message": "Session not found."}), 404

    if not sess["started_at"]:
        conn.execute(
            "UPDATE exam_sessions SET started_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), session_id)
        )
        conn.commit()

    roster = _build_exam_roster(conn, session_id)
    conn.close()

    EXAM_STATE[session_id] = exam_ai.new_session_state()
    EXAM_ROSTER_CACHE[session_id] = roster
    return jsonify({"success": True, "roster_size": len(roster)})


@app.route("/exam/monitor/analyze", methods=["POST"])
@login_required
def exam_monitor_analyze():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    image_data = data.get("image")
    if not session_id or not image_data:
        return jsonify({"success": False, "message": "Missing session or image."}), 400
    session_id = int(session_id)

    state = EXAM_STATE.get(session_id)
    roster = EXAM_ROSTER_CACHE.get(session_id)
    if state is None or roster is None:
        # Server restarted or session unknown to this process — reinit
        # rather than error out, so monitoring can keep running.
        conn = get_db()
        roster = _build_exam_roster(conn, session_id)
        conn.close()
        state = exam_ai.new_session_state()
        EXAM_STATE[session_id] = state
        EXAM_ROSTER_CACHE[session_id] = roster

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Could not read image."}), 400

    if recognizer is None:
        load_recognizer()

    gallery = face_engine.load_gallery(is_teacher=False)
    alerts, boxes, telemetry = exam_ai.analyze_frame(img, state, session_id=session_id, roster=roster, gallery=gallery)

    if alerts:
        conn = get_db()
        now_iso = datetime.now().isoformat(timespec="seconds")
        for a in alerts:
            cur = conn.execute(
                "INSERT INTO exam_malpractice_alerts "
                "(exam_session_id, student_id, usn, student_name, seat_no, alert_type, severity, "
                "detail, confidence, photo_path, video_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, a.get("student_id"), a.get("usn"), a.get("student_name"), a.get("seat_no"),
                 a["type"], a["severity"], a.get("detail", ""), a.get("confidence", 0.0), a.get("photo_path"), a.get("video_path"), now_iso)
            )
            a["alert_id"] = cur.lastrowid

            raw_tid = a.get("track_id")
            parsed_tid = int(str(raw_tid).replace("TRACK_", "")) if (raw_tid is not None and str(raw_tid).replace("TRACK_", "").isdigit()) else None

            conn.execute(
                "INSERT INTO exam_events "
                "(exam_session_id, track_id, student_id, usn, student_name, event_type, status, severity, "
                "duration_seconds, confidence, photo_path, video_path, identity_source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, parsed_tid, a.get("student_id"), a.get("usn"), a.get("student_name"),
                 a["type"], a.get("status", "REVIEW_REQUIRED"), a.get("severity", "medium"),
                 a.get("duration_seconds", config.EVENT_MIN_DURATION), a.get("confidence", 0.0),
                 a.get("photo_path"), a.get("video_path"), a.get("identity_source", "FACE_VERIFIED"), now_iso)
            )
        conn.commit()
        conn.close()
        for a in alerts:
            a["created_at"] = now_iso

    return jsonify({
        "success": True,
        "alerts": alerts,
        "boxes": boxes,
        "telemetry": telemetry,
        "clip_seconds": config.EVIDENCE_TOTAL_CLIP_SECONDS
    })


@app.route("/exam/monitor/alert_video", methods=["POST"])
@login_required
def exam_monitor_alert_video():
    """Receive the short (6-15s) buffered video clip the browser records
    around a medium/high-severity alert (looking_back / phone_use / copying)
    and attach it to that alert's row. The clip is recorded client-side --
    the server only ever sees single JPEG frames per analyze() call, so it
    has no way to produce a clip on its own. Same "supporting context for a
    human invigilator to review" status as the evidence photo -- not proof
    on its own. Same pattern as activeness_alert_video()."""
    session_id = request.form.get("session_id")
    alert_id = request.form.get("alert_id")
    video_file = request.files.get("video")
    if not session_id or not alert_id or not video_file:
        return jsonify({"success": False, "message": "Missing session, alert id, or video."}), 400

    conn = get_db()
    row = conn.execute(
        "SELECT id FROM exam_malpractice_alerts WHERE id = ? AND exam_session_id = ?",
        (alert_id, session_id),
    ).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Alert not found for this session."}), 404

    out_dir = os.path.join(EXAM_PHOTOS_DIR, str(session_id), "videos")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{alert_id}_{int(time.time() * 1000)}.webm"
    video_file.save(os.path.join(out_dir, fname))
    rel_path = f"exam_photos/{session_id}/videos/{fname}"

    conn.execute("UPDATE exam_malpractice_alerts SET video_path = ? WHERE id = ?", (rel_path, alert_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "video_path": rel_path})


@app.route("/exam/monitor/end", methods=["POST"])
@login_required
def exam_monitor_end():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"success": False, "message": "Missing session."}), 400
    session_id = int(session_id)

    conn = get_db()
    conn.execute(
        "UPDATE exam_sessions SET ended_at = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="seconds"), session_id)
    )
    conn.commit()
    conn.close()

    EXAM_STATE.pop(session_id, None)
    EXAM_ROSTER_CACHE.pop(session_id, None)
    return jsonify({"success": True})


@app.route("/exam/sessions/<int:session_id>")
@login_required
def exam_session_detail(session_id):
    conn = get_db()
    sess = conn.execute("""
        SELECT es.*, r.room_name, c.name AS class_name
        FROM exam_sessions es
        JOIN exam_rooms r ON r.id = es.room_id
        LEFT JOIN classes c ON c.id = es.class_id
        WHERE es.id = ?
    """, (session_id,)).fetchone()

    if not sess:
        conn.close()
        flash("Exam session not found.", "danger")
        return redirect(url_for("exam_sessions_list"))

    alerts = conn.execute(
        "SELECT * FROM exam_malpractice_alerts WHERE exam_session_id = ? ORDER BY created_at DESC",
        (session_id,)
    ).fetchall()

    per_student = conn.execute("""
        SELECT usn, student_name, seat_no, COUNT(*) AS alert_count,
               SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) AS high_count
        FROM exam_malpractice_alerts
        WHERE exam_session_id = ? AND student_id IS NOT NULL
        GROUP BY student_id
        ORDER BY alert_count DESC
    """, (session_id,)).fetchall()
    conn.close()

    return render_template(
        "exam_session_detail.html", sess=sess, alerts=alerts, per_student=per_student
    )


@app.route("/exam/sessions/<int:session_id>/delete", methods=["POST"])
@login_required
def exam_session_delete(session_id):
    conn = get_db()
    conn.execute("DELETE FROM exam_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    EXAM_STATE.pop(session_id, None)
    EXAM_ROSTER_CACHE.pop(session_id, None)
    shutil.rmtree(os.path.join(EXAM_PHOTOS_DIR, str(session_id)), ignore_errors=True)
    flash("Exam session deleted.", "info")
    return redirect(url_for("exam_sessions_list"))


# ---------------------------------------------------------------------
# Class Activeness Monitoring
# ---------------------------------------------------------------------
def _make_activeness_identify_fn(session_id):
    """Build the identify_face_fn callback activeness_ai calls periodically
    per track. Reuses the already-trained student LBPH `recognizer` -- the
    same one used for attendance -- restricted to the class this session was
    started for, via the roster cached at start time."""
    roster = ACTIVENESS_CLASS_ROSTER.get(session_id, {})

    def _identify(face_gray_crop):
        if recognizer is None or not roster:
            return None, None
        try:
            face_img = cv2.resize(face_gray_crop, (200, 200))
            label, confidence = recognizer.predict(face_img)
        except cv2.error:
            return None, None
        if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD and label in roster:
            name, _roll_no = roster[label]
            return label, name
        return None, None

    return _identify


def _make_activeness_evidence_fn(session_id):
    """Build the save_evidence_fn callback activeness_ai calls to persist a
    cropped photo the first time a disengagement signal is confirmed for a
    student, alongside the alert -- same pattern as exam_ai's evidence fn."""
    out_dir = os.path.join(ACTIVENESS_PHOTOS_DIR, str(session_id))
    os.makedirs(out_dir, exist_ok=True)

    def save_evidence_fn(img_bgr, box, alert_type, student_id, student_name):
        x, y, bw, bh = box
        pad_x, pad_y = int(bw * 0.4), int(bh * 0.4)
        H, W = img_bgr.shape[:2]
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(W, x + bw + pad_x), min(H, y + bh + pad_y)
        crop = img_bgr[y0:y1, x0:x1]
        if crop is None or crop.size == 0:
            crop = img_bgr
        tag = f"student-{student_id}" if student_id else "unidentified"
        tag = "".join(c for c in tag if c.isalnum() or c in "-_") or "unidentified"
        fname = f"{int(time.time() * 1000)}_{alert_type}_{tag}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, crop)
        return f"activeness_photos/{session_id}/{fname}"
    return save_evidence_fn


@app.route("/attendance/activeness")
@login_required
def activeness_monitor():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()
    conn.close()
    class_id = request.args.get("class_id") or (str(all_classes[0]["id"]) if all_classes else "")
    return render_template(
        "activeness.html",
        classes=all_classes,
        selected_class=str(class_id),
        model_status=exam_ai.get_model_status(),
    )


@app.route("/attendance/activeness/start", methods=["POST"])
@login_required
def activeness_start():
    data = request.get_json(silent=True) or {}
    class_id = data.get("class_id")
    label = (data.get("label") or "").strip()
    if not class_id:
        return jsonify({"success": False, "message": "Missing class."}), 400

    def _do_start():
        with get_db_transaction() as tx_conn:
            cur = tx_conn.execute(
                "INSERT INTO activeness_sessions (class_id, label, started_at) VALUES (?, ?, ?)",
                (class_id, label or None, datetime.now().isoformat(timespec="seconds"))
            )
            sess_id = cur.lastrowid
            roster_data = {}
            for row in tx_conn.execute("SELECT id, name, roll_no, usn FROM students WHERE class_id = ?", (class_id,)).fetchall():
                r_usn = row["usn"] or row["roll_no"]
                roster_data[row["id"]] = {
                    "id": row["id"],
                    "name": row["name"],
                    "roll_no": row["roll_no"],
                    "usn": r_usn,
                    "student_name": row["name"]
                }
            return sess_id, roster_data

    session_id, roster = execute_with_retry(_do_start)

    ACTIVENESS_STATE[session_id] = classroom_ai.new_session_state()
    ACTIVENESS_CLASS_ROSTER[session_id] = roster
    return jsonify({"success": True, "session_id": session_id})


@app.route("/attendance/activeness/analyze", methods=["POST"])
@login_required
def activeness_analyze():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    image_data = data.get("image")
    if not session_id or not image_data:
        return jsonify({"success": False, "message": "Missing session or image."}), 400

    session_id = int(session_id)
    state = ACTIVENESS_STATE.get(session_id)
    if state is None:
        state = classroom_ai.new_session_state()
        ACTIVENESS_STATE[session_id] = state

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Could not read image."}), 400

    roster = ACTIVENESS_CLASS_ROSTER.get(session_id, {})
    gallery = get_gallery() if 'get_gallery' in globals() else None

    # Run multi-modal classroom intelligence analysis
    analysis = classroom_ai.analyze_frame(
        img, state, session_id=session_id, expected_roster=roster, gallery=gallery
    )

    summary = analysis["summary"]
    students = analysis["students"]
    heatmap = analysis["heatmap"]
    boxes = analysis["boxes"]
    alerts = analysis["alerts"]

    now_iso = datetime.now().isoformat(timespec="seconds")

    def _save_activeness_data():
        with get_db_transaction() as tx_conn:
            # 1. Update student activity
            for s in students:
                tid_int = int(s["track_id"].replace("TRACK_", ""))
                tx_conn.execute("""
                    INSERT INTO student_activity (session_id, track_id, student_id, usn, student_name,
                        activity_state, attention_state, facial_expression, id_card_status, uniform_status,
                        seat_status, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, tid_int, s.get("student_id"), s.get("usn"),
                      s.get("name"), s.get("current_activity"), s.get("attention_state"),
                      s.get("facial_expression"), s.get("id_card"), "COMPLIANT",
                      s.get("seat_status"), s.get("confidence", 85.0), now_iso))

                # Backward-compatible sync with activeness_snapshots
                is_active = 1 if s.get("current_activity") in ("ATTENTIVE", "WRITING", "READING", "RAISING_HAND") else 0
                is_away = 1 if s.get("current_activity") in ("LOOKING_AWAY", "LOOKING_BACK") else 0
                is_sleep = 1 if "SLEEP" in s.get("current_activity", "") or "HEAD_DOWN" in s.get("current_activity", "") else 0
                is_phone = 1 if "PHONE" in s.get("current_activity", "") else 0
                band_val = "active" if is_active else ("turned_away" if is_away else ("sleeping" if is_sleep else "phone_use"))

                tx_conn.execute("""
                    INSERT INTO activeness_snapshots
                        (session_id, track_id, student_id, student_name, band, total_scored,
                         active_samples, turned_away_samples, eyes_closed_samples, phone_use_samples, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, track_id) DO UPDATE SET
                        student_id = excluded.student_id,
                        student_name = excluded.student_name,
                        band = excluded.band,
                        total_scored = activeness_snapshots.total_scored + 1,
                        active_samples = activeness_snapshots.active_samples + excluded.active_samples,
                        turned_away_samples = activeness_snapshots.turned_away_samples + excluded.turned_away_samples,
                        eyes_closed_samples = activeness_snapshots.eyes_closed_samples + excluded.eyes_closed_samples,
                        phone_use_samples = activeness_snapshots.phone_use_samples + excluded.phone_use_samples,
                        updated_at = excluded.updated_at
                """, (session_id, tid_int, s.get("student_id"), s.get("name"), band_val, 1,
                      is_active, is_away, is_sleep, is_phone, now_iso))

            # 2. Update alerts
            for a in alerts:
                tid_int = int(a["track_id"].replace("TRACK_", "")) if isinstance(a.get("track_id"), str) else a.get("track_id", 1)
                cur = tx_conn.execute("""
                    INSERT INTO classroom_alerts (session_id, track_id, student_id, usn, student_name,
                        alert_type, severity, detail, confidence, duration_seconds, photo_path, video_path,
                        identity_source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, tid_int, a.get("student_id"), a.get("usn"),
                      a.get("student_name"), a.get("alert_type"), a.get("severity", "medium"),
                      a.get("detail", ""), a.get("confidence", 85.0), a.get("duration_seconds", 3.0),
                      a.get("photo_path", ""), a.get("video_path", ""),
                      a.get("identity_source", "FACE_VERIFIED"), now_iso))
                a["alert_id"] = cur.lastrowid

                # Backward compatibility with activeness_alerts
                tx_conn.execute("""
                    INSERT INTO activeness_alerts
                        (session_id, student_id, student_name, alert_type, detail, photo_path, severity, video_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, a.get("student_id"), a.get("student_name"), a.get("alert_type"),
                      a.get("detail"), a.get("photo_path"), a.get("severity", "low"), a.get("video_path"), now_iso))

            # 3. Log classroom periodic metrics every 5 frames
            if state.get("total_frames_analyzed", 0) % 5 == 0:
                tx_conn.execute("""
                    INSERT INTO classroom_metrics (session_id, timestamp, total_students, present_count,
                        absent_count, unknown_count, activity_index, engagement_level, breakdown_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, now_iso, summary.get("total_students", 0), summary.get("present", 0),
                      summary.get("absent", 0), summary.get("unknown", 0),
                      summary.get("classroom_activity_index", 0.0), summary.get("observable_engagement", "NORMAL"),
                      json.dumps(summary.get("activity_breakdown", {}))))

    execute_with_retry(_save_activeness_data)

    return jsonify({
        "success": True,
        "summary": summary,
        "students": students,
        "heatmap": heatmap,
        "boxes": boxes,
        "alerts": alerts,
        "clip_seconds": 10
    })


@app.route("/attendance/activeness/alert_video", methods=["POST"])
@login_required
def activeness_alert_video():
    """Receive the short (5-10s) buffered video clip the browser records
    around an escalated alert (sleeping / sustained phone use) and attach
    it to that alert's row. The clip itself is recorded client-side --
    the server only ever sees single JPEG frames per analyze() call, so
    it has no way to produce a clip on its own. Same "supporting context
    for a human to review" status as the evidence photo -- not proof."""
    session_id = request.form.get("session_id")
    alert_id = request.form.get("alert_id")
    video_file = request.files.get("video")
    if not session_id or not alert_id or not video_file:
        return jsonify({"success": False, "message": "Missing session, alert id, or video."}), 400

    conn = get_db()
    row = conn.execute(
        "SELECT id FROM activeness_alerts WHERE id = ? AND session_id = ?",
        (alert_id, session_id),
    ).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Alert not found for this session."}), 404

    out_dir = os.path.join(ACTIVENESS_PHOTOS_DIR, str(session_id), "videos")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{alert_id}_{int(time.time() * 1000)}.webm"
    video_file.save(os.path.join(out_dir, fname))
    rel_path = f"activeness_photos/{session_id}/videos/{fname}"

    conn.execute("UPDATE activeness_alerts SET video_path = ? WHERE id = ?", (rel_path, alert_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "video_path": rel_path})


@app.route("/attendance/activeness/end", methods=["POST"])
@login_required
def activeness_end():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"success": False, "message": "Missing session."}), 400

    conn = get_db()
    conn.execute(
        "UPDATE activeness_sessions SET ended_at = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="seconds"), session_id)
    )
    conn.commit()
    conn.close()

    session_id = int(session_id)
    ACTIVENESS_STATE.pop(session_id, None)
    ACTIVENESS_CLASS_ROSTER.pop(session_id, None)
    return jsonify({"success": True})




@app.route("/attendance/activeness/timeline/<string:usn>")
@login_required
def activeness_student_timeline(usn):
    conn = get_db()
    rows = conn.execute("""
        SELECT activity_state, attention_state, timestamp, confidence
        FROM student_activity
        WHERE usn = ?
        ORDER BY timestamp DESC
        LIMIT 50
    """, (usn,)).fetchall()
    conn.close()
    return jsonify({"success": True, "timeline": [dict(r) for r in rows]})


@app.route("/attendance/activeness/predictions/<string:usn>")
@login_required
def activeness_student_predictions(usn):
    conn = get_db()
    rows = conn.execute("""
        SELECT prediction_indicator, confidence, basis_summary, recommendation, created_at
        FROM behavior_predictions
        WHERE usn = ?
        ORDER BY created_at DESC
        LIMIT 10
    """, (usn,)).fetchall()
    conn.close()
    return jsonify({"success": True, "predictions": [dict(r) for r in rows]})

@app.route("/activeness/sessions")
@login_required
def activeness_sessions_list():
    conn = get_db()
    rows = conn.execute("""
        SELECT acs.id, acs.label, acs.started_at, acs.ended_at, c.name AS class_name,
               (SELECT COUNT(*) FROM activeness_snapshots sn WHERE sn.session_id = acs.id) AS students_tracked,
               (SELECT COALESCE(SUM(sn.active_samples), 0) FROM activeness_snapshots sn WHERE sn.session_id = acs.id) AS active_sum,
               (SELECT COALESCE(SUM(sn.total_scored), 0) FROM activeness_snapshots sn WHERE sn.session_id = acs.id) AS scored_sum,
               (SELECT COUNT(*) FROM activeness_alerts al WHERE al.session_id = acs.id) AS alert_count
        FROM activeness_sessions acs
        JOIN classes c ON c.id = acs.class_id
        ORDER BY acs.started_at DESC
        LIMIT 200
    """).fetchall()
    conn.close()
    sessions = []
    for s in rows:
        d = dict(s)
        d["class_active_pct"] = round(100 * d["active_sum"] / d["scored_sum"]) if d["scored_sum"] else None
        sessions.append(d)
    return render_template("activeness_sessions.html", sessions=sessions)


@app.route("/activeness/sessions/<int:session_id>")
@login_required
def activeness_session_detail(session_id):
    conn = get_db()
    sess = conn.execute("""
        SELECT acs.*, c.name AS class_name
        FROM activeness_sessions acs
        JOIN classes c ON c.id = acs.class_id
        WHERE acs.id = ?
    """, (session_id,)).fetchone()

    if not sess:
        conn.close()
        flash("Activeness session not found.", "danger")
        return redirect(url_for("activeness_sessions_list"))

    raw_snapshots = conn.execute("""
        SELECT * FROM activeness_snapshots WHERE session_id = ?
        ORDER BY student_name ASC
    """, (session_id,)).fetchall()

    # Attach display percentages per student -- computed here (not stored)
    # so the formula lives in one place (activeness_ai.percentages).
    snapshots = []
    class_active_sum, class_scored_sum = 0, 0
    for row in raw_snapshots:
        d = dict(row)
        band_counts = {
            "active": d["active_samples"], "turned_away": d["turned_away_samples"],
            "eyes_closed": d["eyes_closed_samples"], "phone_use": d["phone_use_samples"],
        }
        d.update(activeness_ai.percentages(band_counts, d["total_scored"]))
        snapshots.append(d)
        class_active_sum += d["active_samples"]
        class_scored_sum += d["total_scored"]

    class_active_pct = round(100 * class_active_sum / class_scored_sum) if class_scored_sum else None

    raw_alerts = conn.execute("""
        SELECT * FROM activeness_alerts WHERE session_id = ? ORDER BY created_at DESC
    """, (session_id,)).fetchall()
    conn.close()

    alerts = []
    for row in raw_alerts:
        d = dict(row)
        if not d.get("video_path") and d.get("photo_path"):
            event_folder = os.path.dirname(d["photo_path"]).replace("\\", "/")
            cand = os.path.join(BASE_DIR, "static", event_folder, "evidence.mp4")
            if os.path.isfile(cand):
                d["video_path"] = f"{event_folder}/evidence.mp4"
        alerts.append(d)

    return render_template(
        "activeness_session_detail.html",
        sess=sess, snapshots=snapshots, alerts=alerts, class_active_pct=class_active_pct,
        behavior_ai_auto_flag_map=behavior_ai.AUTO_FLAG_TYPE_FOR_BAND,
    )


# ---------------------------------------------------------------------
# Behavioral & Discipline Risk Prediction (Advanced AI Feature 9.1)
# ---------------------------------------------------------------------
@app.route("/behavior")
@login_required
def behavior_dashboard():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()
    class_id = request.args.get("class_id") or (str(all_classes[0]["id"]) if all_classes else "")

    risk_rows = []
    if class_id:
        risk_rows = behavior_ai.compute_class_risk(conn, class_id)

    students = conn.execute(
        "SELECT id, roll_no, name FROM students WHERE class_id = ? ORDER BY roll_no",
        (class_id,),
    ).fetchall() if class_id else []
    conn.close()

    high_count = sum(1 for r in risk_rows if r["level"] == "High")
    medium_count = sum(1 for r in risk_rows if r["level"] == "Medium")

    return render_template(
        "behavior.html",
        classes=all_classes,
        selected_class=str(class_id),
        risk_rows=risk_rows,
        students=students,
        high_count=high_count,
        medium_count=medium_count,
        flag_types=behavior_ai.FLAG_TYPES,
        lookback_days=behavior_ai.LOOKBACK_DAYS,
    )


@app.route("/behavior/student/<int:student_id>")
@login_required
def behavior_student_detail(student_id):
    conn = get_db()
    student = conn.execute(
        "SELECT s.*, c.name AS class_name FROM students s "
        "JOIN classes c ON c.id = s.class_id WHERE s.id = ?",
        (student_id,),
    ).fetchone()
    if not student:
        conn.close()
        flash("Student not found.", "danger")
        return redirect(url_for("behavior_dashboard"))

    risk = behavior_ai.compute_student_risk(conn, student_id)
    flags = conn.execute(
        "SELECT * FROM discipline_flags WHERE student_id = ? ORDER BY created_at DESC",
        (student_id,),
    ).fetchall()
    conn.close()

    return render_template(
        "behavior_student_detail.html",
        student=student,
        risk=risk,
        flags=flags,
        flag_types=behavior_ai.FLAG_TYPES,
        lookback_days=behavior_ai.LOOKBACK_DAYS,
    )


@app.route("/behavior/flag/add", methods=["POST"])
@login_required
def behavior_flag_add():
    student_id = request.form.get("student_id")
    class_id = request.form.get("class_id")
    flag_type = (request.form.get("flag_type") or "").strip()
    severity = request.form.get("severity") or "low"
    note = (request.form.get("note") or "").strip()

    if not student_id or not class_id or not flag_type:
        flash("Please choose a student and an incident type.", "danger")
        return redirect(request.referrer or url_for("behavior_dashboard"))

    if severity not in ("low", "medium", "high"):
        severity = "low"

    conn = get_db()
    conn.execute(
        "INSERT INTO discipline_flags (student_id, class_id, flag_type, severity, note, created_by, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (student_id, class_id, flag_type, severity, note or None,
         session.get("username"), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()

    flash("Incident logged.", "success")
    next_url = request.form.get("next") or url_for("behavior_dashboard", class_id=class_id)
    return redirect(next_url)


@app.route("/behavior/flag/<int:flag_id>/delete", methods=["POST"])
@login_required
def behavior_flag_delete(flag_id):
    conn = get_db()
    row = conn.execute("SELECT student_id, class_id FROM discipline_flags WHERE id = ?", (flag_id,)).fetchone()
    conn.execute("DELETE FROM discipline_flags WHERE id = ?", (flag_id,))
    conn.commit()
    conn.close()

    flash("Incident record removed.", "success")
    if row:
        return redirect(request.referrer or url_for("behavior_student_detail", student_id=row["student_id"]))
    return redirect(url_for("behavior_dashboard"))


# ---------------------------------------------------------------------
# Automatic Data Expiry (Advanced AI Feature 9.4)
# ---------------------------------------------------------------------
@app.route("/settings/data-retention")
@login_required
def data_retention():
    conn = get_db()
    due = data_expiry.find_due_for_expiry(
        conn, has_face_data_fn=lambda sid: face_sample_count(sid) > 0
    )
    recent_log = conn.execute(
        "SELECT * FROM data_expiry_log ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()

    return render_template(
        "data_retention.html",
        due=due,
        recent_log=recent_log,
        graduated_grace_days=data_expiry.GRADUATED_GRACE_DAYS,
        inactivity_days=data_expiry.INACTIVITY_DAYS,
    )


@app.route("/settings/data-retention/run", methods=["POST"])
@login_required
def data_retention_run():
    conn = get_db()
    purged = run_data_expiry_sweep(conn, triggered_by="manual")
    conn.close()

    if purged:
        flash(f"Deleted face photos for {len(purged)} student(s) per retention policy.", "success")
    else:
        flash("Nothing is currently due for deletion.", "info")
    return redirect(url_for("data_retention"))


@app.route("/settings/data-retention/purge/<int:student_id>", methods=["POST"])
@login_required
def data_retention_purge_one(student_id):
    conn = get_db()
    due = data_expiry.find_due_for_expiry(
        conn, has_face_data_fn=lambda sid: face_sample_count(sid) > 0
    )
    candidate = next((c for c in due if c["student_id"] == student_id), None)

    if candidate is None:
        flash("That student isn't currently due for deletion (grace period may not have elapsed yet).", "danger")
    else:
        purge_student_face_data(conn, candidate, triggered_by="manual")
        conn.commit()
        flash(f"Deleted face photos for {candidate['name']}.", "success")

    conn.close()
    return redirect(url_for("data_retention"))


# ---------------------------------------------------------------------
# Attendance marking
# ---------------------------------------------------------------------
@app.route("/attendance", methods=["GET", "POST"])
@login_required
def attendance():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()

    class_id = request.values.get("class_id") or (all_classes[0]["id"] if all_classes else None)
    att_date = request.values.get("att_date") or date.today().isoformat()

    if request.method == "POST":
        student_ids = request.form.getlist("student_id")
        def _do_attendance_save():
            with get_db_transaction() as tx_conn:
                for sid in student_ids:
                    status = request.form.get(f"status_{sid}", "Absent")
                    tx_conn.execute("""
                        INSERT INTO attendance (student_id, class_id, att_date, status)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(student_id, att_date)
                        DO UPDATE SET status = excluded.status
                    """, (sid, class_id, att_date, status))

        execute_with_retry(_do_attendance_save)
        flash(f"Attendance saved for {att_date}.", "success")
        conn.close()
        return redirect(url_for("attendance", class_id=class_id, att_date=att_date))

    roster = []
    if class_id:
        roster = conn.execute("""
            SELECT s.id, s.roll_no, s.name,
                   COALESCE(a.status, 'Present') AS status
            FROM students s
            LEFT JOIN attendance a
                ON a.student_id = s.id AND a.att_date = ?
            WHERE s.class_id = ?
            ORDER BY s.roll_no
        """, (att_date, class_id)).fetchall()

    conn.close()
    return render_template(
        "attendance.html",
        classes=all_classes,
        roster=roster,
        selected_class=str(class_id) if class_id else "",
        att_date=att_date
    )


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------
@app.route("/reports")
@login_required
def reports():
    conn = get_db()
    all_classes = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()

    class_id = request.args.get("class_id", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")

    query = """
        SELECT s.roll_no, s.name, c.name AS class_name,
               SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) AS present_count,
               SUM(CASE WHEN a.status='Absent' THEN 1 ELSE 0 END) AS absent_count,
               SUM(CASE WHEN a.status='Late' THEN 1 ELSE 0 END) AS late_count,
               COUNT(a.id) AS total_marked
        FROM students s
        JOIN classes c ON c.id = s.class_id
        LEFT JOIN attendance a ON a.student_id = s.id
    """
    conditions = []
    params = []

    if class_id:
        conditions.append("s.class_id = ?")
        params.append(class_id)
    if start_date:
        conditions.append("(a.att_date IS NULL OR a.att_date >= ?)")
        params.append(start_date)
    if end_date:
        conditions.append("(a.att_date IS NULL OR a.att_date <= ?)")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " GROUP BY s.id ORDER BY c.name, s.roll_no"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    report_data = []
    for r in rows:
        total = r["total_marked"] or 0
        pct = round((r["present_count"] / total) * 100, 1) if total > 0 else 0.0
        report_data.append({**dict(r), "percentage": pct})

    return render_template(
        "reports.html",
        classes=all_classes,
        report_data=report_data,
        selected_class=class_id,
        start_date=start_date,
        end_date=end_date
    )


@app.route("/reports/export")
@login_required
def export_report():
    conn = get_db()
    class_id = request.args.get("class_id", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")

    query = """
        SELECT s.roll_no, s.name, c.name AS class_name, a.att_date, a.status
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        JOIN classes c ON c.id = a.class_id
    """
    conditions = []
    params = []
    if class_id:
        conditions.append("s.class_id = ?")
        params.append(class_id)
    if start_date:
        conditions.append("a.att_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.att_date <= ?")
        params.append(end_date)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY a.att_date, c.name, s.roll_no"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    def generate():
        yield "Roll No,Name,Class,Date,Status\n"
        for r in rows:
            yield f'{r["roll_no"]},{r["name"]},{r["class_name"]},{r["att_date"]},{r["status"]}\n'

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_report.csv"}
    )


# ---------------------------------------------------------------------

@app.route("/api/exam/event/<int:alert_id>")
@login_required
def api_exam_event(alert_id):
    conn = get_db()
    row = conn.execute("""
        SELECT a.*, es.exam_name, r.room_name
        FROM exam_malpractice_alerts a
        JOIN exam_sessions es ON es.id = a.exam_session_id
        JOIN exam_rooms r ON r.id = es.room_id
        WHERE a.id = ?
    """, (alert_id,)).fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "message": "Event not found"}), 404

    # Locate evidence folder for multi-photo evidence (Item 11, 12)
    session_id = row["exam_session_id"]
    event_folder = os.path.join(config.EXAM_PHOTOS_DIR, str(session_id), f"event_{alert_id}")
    
    photos = {}
    for img_name in ["before.jpg", "event_start.jpg", "best_event.jpg", "event_end.jpg", "after.jpg", "student_crop.jpg", "face_crop.jpg", "spatial_neighbor.jpg", "full_frame_before.jpg", "full_frame_event.jpg"]:
        if os.path.exists(os.path.join(event_folder, img_name)):
            photos[img_name.split(".")[0]] = f"exam_photos/{session_id}/event_{alert_id}/{img_name}"

    video_rel = f"exam_photos/{session_id}/event_{alert_id}/evidence.mp4"
    if os.path.exists(os.path.join(event_folder, "evidence.mp4")):
        vid_path = video_rel
    else:
        vid_path = row["video_path"]

    evidence = {
        "event_id": row["id"],
        "exam_name": row["exam_name"],
        "room_name": row["room_name"],
        "student_id": row["student_id"],
        "student_name": row["student_name"],
        "usn": row["usn"],
        "seat_no": row["seat_no"],
        "alert_type": row["alert_type"],
        "severity": row["severity"],
        "detail": row["detail"],
        "confidence": row["confidence"],
        "created_at": row["created_at"],
        "photo_path": row["photo_path"],
        "video_path": vid_path,
        "photos": photos if photos else {
            "event": row["photo_path"]
        }
    }
    return jsonify({"success": True, "event": evidence})

if __name__ == "__main__":
    init_db()
    ensure_face_dirs()
    load_recognizer()
    load_teacher_recognizer()
    # use_reloader=False prevents duplicate process initialization competing for SQLite locks
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "false").lower() in ("true", "1")
    app.run(debug=True, use_reloader=use_reloader)
