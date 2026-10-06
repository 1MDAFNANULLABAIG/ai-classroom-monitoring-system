"""
REST API and WebSocket Engine for Smart Classroom Monitoring
============================================================
Extends the attendance and malpractice system with modern RESTful APIs
and high-frequency WebSockets for React/Next.js frontends.
"""

import os
import io
import time
import json
import csv
import base64
import sqlite3
import glob
from collections import deque
from datetime import datetime, date, timedelta
from functools import wraps

from flask import Blueprint, request, jsonify, Response, current_app
import cv2
import numpy as np
import jwt

import config
import face_quality
import face_engine
import classroom_ai
import tracker
import yolo_detector

JWT_SECRET = os.environ.get("JWT_SECRET", "super-secret-classroom-jwt-key-2026-ultra-secure-32chars")
JWT_ALGORITHM = "HS256"

api_bp = Blueprint("api_v2", __name__, url_prefix="/api")

# In-memory runtime state for live classroom sessions
_ACTIVE_CLASSROOM_SESSIONS = {}  # class_id -> session_dict
_CLASSROOM_TRACKERS = {}         # class_id -> ClassroomTracker
_LIVE_PROFILERS = {}             # class_id -> dict of timing windows


def create_token(user_id, username, role):
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=7),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


def get_authenticated_user():
    """Extract user from Authorization header or session."""
    import app
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        data = decode_token(token)
        if data:
            conn = app.get_db()
            user = conn.execute("SELECT id, username, role FROM users WHERE id = ?", (data["user_id"],)).fetchone()
            conn.close()
            if user:
                return dict(user)

    # Fallback to session
    from flask import session
    if "user_id" in session:
        conn = app.get_db()
        user = conn.execute("SELECT id, username, role FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        conn.close()
        if user:
            return dict(user)
    return None


def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_authenticated_user()
        if not user:
            return jsonify({"success": False, "error": "Unauthorized. Please log in."}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------
# 1. Health & AI Status
# ---------------------------------------------------------------------
@api_bp.route("/health", methods=["GET"])
def health_check():
    import app
    yolo_ready = yolo_detector.get_yolo_model() is not None
    sf_ready = os.path.exists(config.SFACE_MODEL_PATH)
    yn_ready = os.path.exists(config.YUNET_MODEL_PATH)
    mp_status = classroom_ai.get_model_status()

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_engine": {
            "yolo_ready": yolo_ready,
            "yunet_ready": yn_ready,
            "sface_ready": sf_ready,
            "mediapipe_ready": mp_status["mediapipe_installed"],
            "pose_ready": mp_status["pose_landmarker_ready"],
            "face_mesh_ready": mp_status["face_landmarker_ready"],
        }
    })


# ---------------------------------------------------------------------
# 2. Authentication Routes
# ---------------------------------------------------------------------
@api_bp.route("/auth/login", methods=["POST"])
def auth_login():
    import app
    from werkzeug.security import check_password_hash
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required."}), 400

    conn = app.get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        token = create_token(user["id"], user["username"], user["role"])
        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"]
            }
        })
    return jsonify({"success": False, "error": "Invalid username or password."}), 401


@api_bp.route("/auth/me", methods=["GET"])
def auth_me():
    user = get_authenticated_user()
    if not user:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    return jsonify({"success": True, "user": user})


# ---------------------------------------------------------------------
# 3. Classes Management
# ---------------------------------------------------------------------
@api_bp.route("/classes", methods=["GET"])
def get_classes():
    import app
    conn = app.get_db()
    classes = conn.execute("""
        SELECT c.*, 
               (SELECT COUNT(*) FROM students s WHERE s.class_id = c.id) as student_count
        FROM classes c
        ORDER BY c.name
    """).fetchall()
    conn.close()
    return jsonify({"success": True, "classes": [dict(c) for c in classes]})


@api_bp.route("/classes", methods=["POST"])
@api_login_required
def create_class():
    import app
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    start_time = data.get("start_time", "08:30")
    end_time = data.get("end_time", "16:00")
    department = data.get("department", "ISE")
    semester = data.get("semester", 5)

    if not name:
        return jsonify({"success": False, "error": "Class name is required."}), 400

    app.ensure_db_initialized()
    conn = app.get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO classes (name, start_time, end_time, department, semester) VALUES (?, ?, ?, ?, ?)",
            (name, start_time, end_time, department, semester)
        )
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return jsonify({"success": True, "id": new_id, "message": "Class created successfully."})
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400


# ---------------------------------------------------------------------
# 4. Students Management & Face Enrollment
# ---------------------------------------------------------------------
@api_bp.route("/students", methods=["GET"])
def get_students():
    import app
    class_id = request.args.get("class_id")
    search = request.args.get("search", "").strip()

    conn = app.get_db()
    query = """
        SELECT s.*, c.name as class_name
        FROM students s
        LEFT JOIN classes c ON c.id = s.class_id
        WHERE 1=1
    """
    params = []
    if class_id:
        query += " AND s.class_id = ?"
        params.append(class_id)
    if search:
        query += " AND (s.name LIKE ? OR s.roll_no LIKE ? OR s.usn LIKE ?)"
        wild = f"%{search}%"
        params.extend([wild, wild, wild])

    query += " ORDER BY s.roll_no, s.name"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    students = []
    for r in rows:
        st = dict(r)
        count = app.face_sample_count(st["id"])
        st["sample_count"] = count
        st["face_enrolled"] = (count >= config.SAMPLES_PER_STUDENT)
        students.append(st)

    return jsonify({"success": True, "students": students})


@api_bp.route("/students", methods=["POST"])
@api_login_required
def create_student():
    import app
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    roll_no = data.get("roll_no", "").strip()
    usn = data.get("usn", "").strip() or roll_no
    class_id = data.get("class_id")
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()
    department = data.get("department", "ISE").strip()
    semester = data.get("semester", 5)
    section = data.get("section", "A").strip()
    admission_year = data.get("admission_year", "2023").strip()

    if not name or not roll_no or not class_id:
        return jsonify({"success": False, "error": "Name, Roll No/USN, and Class are required."}), 400

    app.ensure_db_initialized()
    conn = app.get_db()
    try:
        cur = conn.cursor()
        cls_row = cur.execute("SELECT id FROM classes WHERE id = ?", (class_id,)).fetchone()
        if not cls_row:
            # Fallback to first available class if an invalid ID was passed
            fallback_cls = cur.execute("SELECT id FROM classes ORDER BY id ASC LIMIT 1").fetchone()
            if fallback_cls:
                class_id = fallback_cls["id"]
            else:
                conn.close()
                return jsonify({"success": False, "error": f"Class with ID {class_id} not found."}), 400

        cur.execute(
            """INSERT INTO students 
               (name, roll_no, usn, class_id, email, phone, department, current_semester, section, admission_year, status) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
            (name, roll_no, usn, class_id, email, phone, department, semester, section, admission_year)
        )
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return jsonify({"success": True, "id": new_id, "message": "Student registered successfully."})
    except sqlite3.IntegrityError as ie:
        conn.rollback()
        conn.close()
        err_msg = str(ie).lower()
        if "foreign key" in err_msg:
            return jsonify({"success": False, "error": "Selected class does not exist."}), 400
        return jsonify({"success": False, "error": "Roll Number or USN already exists."}), 409
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400


@api_bp.route("/students/<int:student_id>", methods=["GET"])
def get_student_detail(student_id):
    import app
    conn = app.get_db()
    student = conn.execute("""
        SELECT s.*, c.name as class_name
        FROM students s
        LEFT JOIN classes c ON c.id = s.class_id
        WHERE s.id = ?
    """, (student_id,)).fetchone()

    if not student:
        conn.close()
        return jsonify({"success": False, "error": "Student not found."}), 404

    st = dict(student)
    st["sample_count"] = app.face_sample_count(student_id)
    st["face_enrolled"] = (st["sample_count"] >= config.SAMPLES_PER_STUDENT)

    # Attendance stats
    att_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_days,
            SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present_days,
            SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) as absent_days
        FROM attendance
        WHERE student_id = ?
    """, (student_id,)).fetchone()
    st["attendance_stats"] = dict(att_stats) if att_stats else {"total_days": 0, "present_days": 0, "absent_days": 0}

    # Recent alerts
    recent_alerts = conn.execute("""
        SELECT * FROM activeness_alerts 
        WHERE student_id = ? 
        ORDER BY created_at DESC LIMIT 10
    """, (student_id,)).fetchall()
    st["recent_alerts"] = [dict(a) for a in recent_alerts]

    conn.close()
    return jsonify({"success": True, "student": st})


@api_bp.route("/students/<int:student_id>/enroll-frame", methods=["POST"])
@api_login_required
def enroll_student_frame(student_id):
    """
    Accepts real webcam frame, runs YuNet detection & quality check,
    saves sample and extracts 128-D SFace embedding.
    Auto-trains when 50 samples are reached.
    """
    import app
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "error": "No image data received."}), 400

    img = app.decode_base64_image(image_b64)
    if img is None:
        return jsonify({"success": False, "error": "Invalid image format."}), 400

    # 1. Detection
    detections = face_engine.detect_faces(img)
    if len(detections) == 0:
        return jsonify({"success": False, "error": "No face detected. Please face the camera."}), 422
    if len(detections) > 1:
        return jsonify({"success": False, "error": "Multiple faces detected. Ensure only one person is in view."}), 422

    det = detections[0]
    # 2. Quality assessment
    quality = face_quality.assess_face_quality(img, det["box"], det["landmarks"], detection_conf=det["score"])
    if not quality["passed"]:
        reason = quality["reasons"][0] if quality["reasons"] else "Face quality insufficient"
        return jsonify({
            "success": False,
            "error": reason,
            "quality_score": quality["quality_score"],
            "pose": quality["pose"]
        }), 422

    # 3. Save sample image
    app.ensure_face_dirs()
    student_dir = os.path.join(config.DATASET_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)

    curr_count = app.face_sample_count(student_id)
    next_index = curr_count + 1

    x, y, w, h = det["box"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], (200, 200))
    cv2.imwrite(os.path.join(student_dir, f"{next_index}.jpg"), face_img)

    # 4. Extract SFace embedding and store in DB
    _, emb = face_engine.extract_embedding(img, det)
    conn = app.get_db()
    now_iso = datetime.now().isoformat(timespec="seconds")
    if emb is not None:
        conn.execute(
            "INSERT INTO face_embeddings (person_type, person_id, pose_tag, quality_score, embedding_blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("student", student_id, quality["pose"], quality["quality_score"], emb.tobytes(), now_iso)
        )
    conn.execute("UPDATE students SET face_deleted_at = NULL WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()

    # 5. Check if training threshold reached
    auto_trained = False
    if next_index >= config.SAMPLES_PER_STUDENT:
        # Trigger background train & gallery update
        try:
            train_student_models_internal()
            auto_trained = True
        except Exception as e:
            print(f"[warning] Auto-train error: {e}")

    return jsonify({
        "success": True,
        "count": next_index,
        "required": config.SAMPLES_PER_STUDENT,
        "pose": quality["pose"],
        "quality_score": quality["quality_score"],
        "auto_trained": auto_trained,
        "message": f"Sample {next_index}/{config.SAMPLES_PER_STUDENT} captured ({quality['pose']})."
    })


def train_student_models_internal():
    """Trains LBPH student recognizer and reloads SFace gallery."""
    import app
    app.ensure_face_dirs()
    faces, labels = [], []

    for entry in os.listdir(config.DATASET_DIR):
        student_dir = os.path.join(config.DATASET_DIR, entry)
        if not os.path.isdir(student_dir):
            continue
        try:
            student_id = int(entry)
        except ValueError:
            continue
        for img_path in glob.glob(os.path.join(student_dir, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(student_id)

    if len(faces) >= 2 and len(set(labels)) >= 1:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(faces, np.array(labels))
        model.write(config.TRAINER_FILE)
        app.load_recognizer()
        face_engine.load_gallery(is_teacher=False, force_reload=True)
        return True
    return False


@api_bp.route("/students/<int:student_id>/train", methods=["POST"])
@api_login_required
def train_student_model_api(student_id):
    success = train_student_models_internal()
    if success:
        return jsonify({"success": True, "message": "Face recognition models trained and refreshed."})
    return jsonify({"success": False, "error": "Insufficient face samples across students to train."}), 400


# ---------------------------------------------------------------------
# 5. Teachers Management & Face Enrollment
# ---------------------------------------------------------------------
@api_bp.route("/teachers", methods=["GET"])
def get_teachers():
    import app
    conn = app.get_db()
    rows = conn.execute("SELECT * FROM teachers ORDER BY name").fetchall()
    conn.close()

    teachers = []
    for r in rows:
        t = dict(r)
        count = app.face_sample_count(t["id"], base_dir=config.TEACHER_DATASET_DIR)
        t["sample_count"] = count
        t["face_enrolled"] = (count >= config.SAMPLES_PER_TEACHER)
        teachers.append(t)

    return jsonify({"success": True, "teachers": teachers})


@api_bp.route("/teachers", methods=["POST"])
@api_login_required
def create_teacher():
    import app
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    teacher_code = data.get("teacher_code", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    if not name or not teacher_code:
        return jsonify({"success": False, "error": "Name and Teacher Code are required."}), 400

    conn = app.get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO teachers (name, teacher_code, email, phone) VALUES (?, ?, ?, ?)",
            (name, teacher_code, email, phone)
        )
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return jsonify({"success": True, "id": new_id, "message": "Teacher registered successfully."})
    except sqlite3.IntegrityError:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "error": "Teacher Code already exists."}), 409
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400


@api_bp.route("/teachers/<int:teacher_id>/enroll-frame", methods=["POST"])
@api_login_required
def enroll_teacher_frame(teacher_id):
    """Enroll real webcam face frame for teacher."""
    import app
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "error": "No image data received."}), 400

    img = app.decode_base64_image(image_b64)
    if img is None:
        return jsonify({"success": False, "error": "Invalid image format."}), 400

    detections = face_engine.detect_faces(img)
    if len(detections) == 0:
        return jsonify({"success": False, "error": "No face detected. Please face the camera."}), 422
    if len(detections) > 1:
        return jsonify({"success": False, "error": "Multiple faces detected."}), 422

    det = detections[0]
    quality = face_quality.assess_face_quality(img, det["box"], det["landmarks"], detection_conf=det["score"])
    if not quality["passed"]:
        reason = quality["reasons"][0] if quality["reasons"] else "Face quality insufficient"
        return jsonify({
            "success": False,
            "error": reason,
            "quality_score": quality["quality_score"],
            "pose": quality["pose"]
        }), 422

    app.ensure_face_dirs()
    teacher_dir = os.path.join(config.TEACHER_DATASET_DIR, str(teacher_id))
    os.makedirs(teacher_dir, exist_ok=True)

    curr_count = app.face_sample_count(teacher_id, base_dir=config.TEACHER_DATASET_DIR)
    next_index = curr_count + 1

    x, y, w, h = det["box"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], (200, 200))
    cv2.imwrite(os.path.join(teacher_dir, f"{next_index}.jpg"), face_img)

    _, emb = face_engine.extract_embedding(img, det)
    conn = app.get_db()
    now_iso = datetime.now().isoformat(timespec="seconds")
    if emb is not None:
        conn.execute(
            "INSERT INTO face_embeddings (person_type, person_id, pose_tag, quality_score, embedding_blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("teacher", teacher_id, quality["pose"], quality["quality_score"], emb.tobytes(), now_iso)
        )
    conn.commit()
    conn.close()

    auto_trained = False
    if next_index >= config.SAMPLES_PER_TEACHER:
        try:
            train_teacher_models_internal()
            auto_trained = True
        except Exception:
            pass

    return jsonify({
        "success": True,
        "count": next_index,
        "required": config.SAMPLES_PER_TEACHER,
        "pose": quality["pose"],
        "quality_score": quality["quality_score"],
        "auto_trained": auto_trained,
        "message": f"Sample {next_index}/{config.SAMPLES_PER_TEACHER} captured ({quality['pose']})."
    })


def train_teacher_models_internal():
    import app
    import glob
    app.ensure_face_dirs()
    faces, labels = [], []

    for entry in os.listdir(config.TEACHER_DATASET_DIR):
        teacher_dir = os.path.join(config.TEACHER_DATASET_DIR, entry)
        if not os.path.isdir(teacher_dir):
            continue
        try:
            teacher_id = int(entry)
        except ValueError:
            continue
        for img_path in glob.glob(os.path.join(teacher_dir, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(teacher_id)

    if len(faces) >= 2 and len(set(labels)) >= 1:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(faces, np.array(labels))
        model.write(config.TEACHER_TRAINER_FILE)
        app.load_teacher_recognizer()
        face_engine.load_gallery(is_teacher=True, force_reload=True)
        return True
    return False


@api_bp.route("/teachers/<int:teacher_id>/train", methods=["POST"])
@api_login_required
def train_teacher_model_api(teacher_id):
    success = train_teacher_models_internal()
    if success:
        return jsonify({"success": True, "message": "Teacher face recognition model updated."})
    return jsonify({"success": False, "error": "Insufficient teacher face samples to train."}), 400


# ---------------------------------------------------------------------
# 6. Timetable Management & Timetable Verification
# ---------------------------------------------------------------------
@api_bp.route("/timetable", methods=["GET"])
def get_timetable():
    import app
    class_id = request.args.get("class_id")
    conn = app.get_db()
    query = """
        SELECT tt.*, c.name as class_name, t.name as teacher_name
        FROM timetable tt
        LEFT JOIN classes c ON c.id = tt.class_id
        LEFT JOIN teachers t ON t.id = tt.teacher_id
        WHERE 1=1
    """
    params = []
    if class_id:
        query += " AND tt.class_id = ?"
        params.append(class_id)
    query += " ORDER BY tt.start_time, tt.period_number"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify({"success": True, "timetable": [dict(r) for r in rows]})


@api_bp.route("/timetable", methods=["POST"])
@api_login_required
def create_timetable_period():
    import app
    data = request.get_json(silent=True) or {}
    class_id = data.get("class_id")
    period_number = data.get("period_number", 1)
    period_type = data.get("period_type", "class")
    subject = data.get("subject", "").strip()
    teacher_id = data.get("teacher_id")
    start_time = data.get("start_time", "").strip()
    end_time = data.get("end_time", "").strip()

    if not class_id or not start_time or not end_time or not subject:
        return jsonify({"success": False, "error": "Class, Subject, Start Time, and End Time are required."}), 400

    conn = app.get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO timetable (class_id, period_number, period_type, subject, teacher_id, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (class_id, period_number, period_type, subject, teacher_id, start_time, end_time))
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return jsonify({"success": True, "id": new_id, "message": "Period scheduled successfully."})
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400


@api_bp.route("/timetable/<int:period_id>", methods=["DELETE"])
@api_login_required
def delete_timetable_period(period_id):
    import app
    conn = app.get_db()
    conn.execute("DELETE FROM timetable WHERE id = ?", (period_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Period removed."})


@api_bp.route("/timetable/current", methods=["GET"])
def get_current_timetable_period():
    """
    Evaluates current system time against scheduled timetable.
    Returns whether class is scheduled right now, the active period,
    and teacher information.
    """
    import app
    class_id = request.args.get("class_id")
    if not class_id:
        return jsonify({"success": False, "error": "class_id required"}), 400

    now = datetime.now()
    now_str = now.strftime("%H:%M")
    conn = app.get_db()

    # Find matching period in timetable for this class
    period = conn.execute("""
        SELECT tt.*, c.name as class_name, t.name as teacher_name
        FROM timetable tt
        LEFT JOIN classes c ON c.id = tt.class_id
        LEFT JOIN teachers t ON t.id = tt.teacher_id
        WHERE tt.class_id = ? AND tt.start_time <= ? AND tt.end_time >= ?
        LIMIT 1
    """, (class_id, now_str, now_str)).fetchone()

    upcoming = None
    if not period:
        upcoming = conn.execute("""
            SELECT tt.*, c.name as class_name, t.name as teacher_name
            FROM timetable tt
            LEFT JOIN classes c ON c.id = tt.class_id
            LEFT JOIN teachers t ON t.id = tt.teacher_id
            WHERE tt.class_id = ? AND tt.start_time > ?
            ORDER BY tt.start_time ASC
            LIMIT 1
        """, (class_id, now_str)).fetchone()

    conn.close()

    return jsonify({
        "success": True,
        "current_time": now_str,
        "is_active_period": period is not None,
        "active_period": dict(period) if period else None,
        "upcoming_period": dict(upcoming) if upcoming else None,
    })


# ---------------------------------------------------------------------
# 7. Classroom Live Session Controls
# ---------------------------------------------------------------------
@api_bp.route("/classroom/status", methods=["GET"])
def get_classroom_status():
    class_id = request.args.get("class_id")
    if not class_id:
        return jsonify({"success": False, "error": "class_id is required."}), 400

    cid = int(class_id)
    session_data = _ACTIVE_CLASSROOM_SESSIONS.get(cid)

    if not session_data:
        return jsonify({
            "success": True,
            "is_active": False,
            "session": None
        })

    import app
    conn = app.get_db()
    # Compute live counts
    total_students = conn.execute("SELECT COUNT(*) FROM students WHERE class_id = ?", (cid,)).fetchone()[0]
    present_count = len(session_data.get("present_students", set()))
    absent_count = max(0, total_students - present_count)
    pct = round((present_count / max(1, total_students)) * 100, 1)
    conn.close()

    return jsonify({
        "success": True,
        "is_active": True,
        "session": {
            "session_id": session_data["session_id"],
            "class_id": cid,
            "started_at": session_data["started_at"],
            "subject": session_data.get("subject", "Classroom Session"),
            "teacher_confirmed": session_data.get("teacher_confirmed", False),
            "teacher_name": session_data.get("teacher_name", "Unassigned"),
            "total_students": total_students,
            "present_count": present_count,
            "absent_count": absent_count,
            "attendance_percentage": pct,
            "alerts_count": len(session_data.get("alerts", [])),
        }
    })


@api_bp.route("/classroom/start", methods=["POST"])
@api_login_required
def start_classroom_session():
    """
    Starts live monitoring session:
    1. Verifies timetable schedule for current time.
    2. Verifies teacher requirement.
    3. Initializes session state & tracking.
    """
    import app
    data = request.get_json(silent=True) or {}
    class_id = data.get("class_id")
    subject = data.get("subject", "").strip()
    force_start = data.get("force_start", False)  # Admin bypass for testing/overrides

    if not class_id:
        return jsonify({"success": False, "error": "Class ID is required."}), 400

    cid = int(class_id)
    now = datetime.now()
    now_str = now.strftime("%H:%M")
    today_str = now.date().isoformat()

    conn = app.get_db()

    # Timetable verification
    current_period = conn.execute("""
        SELECT tt.*, t.name as teacher_name
        FROM timetable tt
        LEFT JOIN teachers t ON t.id = tt.teacher_id
        WHERE tt.class_id = ? AND tt.start_time <= ? AND tt.end_time >= ?
        LIMIT 1
    """, (cid, now_str, now_str)).fetchone()

    if not current_period and not force_start:
        conn.close()
        return jsonify({
            "success": False,
            "error": f"Timetable Verification: No scheduled class found for this class at {now_str}. Start can be authorized with admin override.",
            "code": "TIMETABLE_NOT_SCHEDULED"
        }), 400

    period_dict = dict(current_period) if current_period else {}
    subj = subject or period_dict.get("subject", "Classroom Intelligence")
    teacher_id = period_dict.get("teacher_id")
    teacher_name = period_dict.get("teacher_name") or "Teacher"

    # Create session record in activeness_sessions
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO activeness_sessions (class_id, label, started_at)
        VALUES (?, ?, ?)
    """, (cid, f"{subj} - {today_str}", now.isoformat()))
    session_id = cur.lastrowid

    # Also record in period_sessions if timetable_id is available
    period_session_id = None
    if period_dict.get("id"):
        existing_ps = conn.execute(
            "SELECT id FROM period_sessions WHERE timetable_id = ? AND session_date = ?",
            (period_dict["id"], today_str)
        ).fetchone()
        if existing_ps:
            period_session_id = existing_ps["id"]
            conn.execute("""
                UPDATE period_sessions 
                SET started_at = ?, ended_at = NULL, teacher_id = ?, teacher_status = 'Present'
                WHERE id = ?
            """, (now.isoformat(), teacher_id, period_session_id))
        else:
            cur.execute("""
                INSERT INTO period_sessions (class_id, timetable_id, session_date, teacher_id, started_at, teacher_status, auto_marked)
                VALUES (?, ?, ?, ?, ?, 'Present', 0)
            """, (cid, period_dict["id"], today_str, teacher_id, now.isoformat()))
            period_session_id = cur.lastrowid

    conn.commit()
    conn.close()

    # Initialize runtime session
    session_info = {
        "session_id": session_id,
        "period_session_id": period_session_id,
        "class_id": cid,
        "subject": subj,
        "timetable_id": period_dict.get("id"),
        "teacher_id": teacher_id,
        "teacher_name": teacher_name,
        "teacher_confirmed": bool(teacher_id is not None),
        "started_at": now.isoformat(),
        "present_students": set(),
        "alerts": [],
        "ai_state": classroom_ai.new_session_state()
    }
    _ACTIVE_CLASSROOM_SESSIONS[cid] = session_info
    _CLASSROOM_TRACKERS[cid] = tracker.ClassroomTracker()

    return jsonify({
        "success": True,
        "message": f"Classroom session started for {subj}.",
        "session": {
            "session_id": session_id,
            "class_id": cid,
            "subject": subj,
            "teacher_name": teacher_name,
            "started_at": now.isoformat(),
        }
    })


@api_bp.route("/classroom/stop", methods=["POST"])
@api_login_required
def stop_classroom_session():
    """
    Stops live monitoring session:
    Marks absent students in the database who were never confirmed present,
    finalizes attendance and session rows.
    """
    import app
    data = request.get_json(silent=True) or {}
    class_id = data.get("class_id")

    if not class_id:
        return jsonify({"success": False, "error": "Class ID is required."}), 400

    cid = int(class_id)
    session_info = _ACTIVE_CLASSROOM_SESSIONS.pop(cid, None)

    if not session_info:
        return jsonify({"success": False, "error": "No active session for this class."}), 404

    now_iso = datetime.now().isoformat()
    today_str = datetime.now().date().isoformat()
    session_id = session_info["session_id"]
    present_ids = session_info.get("present_students", set())

    conn = app.get_db()
    # Mark session ended
    conn.execute("UPDATE activeness_sessions SET ended_at = ? WHERE id = ?", (now_iso, session_id))
    if session_info.get("timetable_id"):
        conn.execute(
            "UPDATE period_sessions SET ended_at = ? WHERE class_id = ? AND session_date = ? AND ended_at IS NULL",
            (now_iso, cid, today_str)
        )

    # Get all students enrolled in this class
    class_students = conn.execute("SELECT id FROM students WHERE class_id = ?", (cid,)).fetchall()
    absent_count = 0

    for st in class_students:
        sid = st["id"]
        if sid not in present_ids:
            # Check if student was already marked
            existing = conn.execute(
                "SELECT id FROM attendance WHERE student_id = ? AND class_id = ? AND att_date = ?",
                (sid, cid, today_str)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO attendance (student_id, class_id, att_date, status) VALUES (?, ?, ?, 'Absent')",
                    (sid, cid, today_str)
                )
                absent_count += 1

            # Also in period_attendance if this session is linked to a timetable period
            p_sid = session_info.get("period_session_id")
            if p_sid:
                p_existing = conn.execute(
                    "SELECT id FROM period_attendance WHERE session_id = ? AND student_id = ?",
                    (p_sid, sid)
                ).fetchone()
                if not p_existing:
                    conn.execute(
                        "INSERT INTO period_attendance (session_id, student_id, status, marked_at) VALUES (?, ?, 'Absent', ?)",
                        (p_sid, sid, now_iso)
                    )

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": "Classroom session finalized successfully.",
        "summary": {
            "session_id": session_id,
            "total_students": len(class_students),
            "present_count": len(present_ids),
            "absent_count": absent_count,
            "ended_at": now_iso
        }
    })


# ---------------------------------------------------------------------
# 8. Attendance Records & CSV Export
# ---------------------------------------------------------------------
@api_bp.route("/attendance", methods=["GET"])
def get_attendance():
    import app
    class_id = request.args.get("class_id")
    att_date = request.args.get("date") or date.today().isoformat()
    status_filter = request.args.get("status")

    conn = app.get_db()
    query = """
        SELECT a.id, a.att_date, a.status, 
               s.id as student_id, s.name as student_name, s.roll_no, s.usn,
               c.name as class_name
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        JOIN classes c ON c.id = a.class_id
        WHERE 1=1
    """
    params = []
    if class_id:
        query += " AND a.class_id = ?"
        params.append(class_id)
    if att_date:
        query += " AND a.att_date = ?"
        params.append(att_date)
    if status_filter:
        query += " AND LOWER(a.status) = LOWER(?)"
        params.append(status_filter)

    query += " ORDER BY s.roll_no ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    return jsonify({"success": True, "attendance": [dict(r) for r in rows]})


@api_bp.route("/attendance/export", methods=["GET"])
def export_attendance_csv():
    """Generates real downloadable CSV attendance report."""
    import app
    class_id = request.args.get("class_id")
    att_date = request.args.get("date") or date.today().isoformat()

    conn = app.get_db()
    query = """
        SELECT a.att_date, s.roll_no, s.usn, s.name as student_name, c.name as class_name, a.status
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        JOIN classes c ON c.id = a.class_id
        WHERE 1=1
    """
    params = []
    if class_id:
        query += " AND a.class_id = ?"
        params.append(class_id)
    if att_date:
        query += " AND a.att_date = ?"
        params.append(att_date)

    query += " ORDER BY s.roll_no ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Roll Number", "USN", "Student Name", "Class", "Attendance Status"])
    for r in rows:
        writer.writerow([r["att_date"], r["roll_no"], r["usn"], r["student_name"], r["class_name"], r["status"].upper()])

    output.seek(0)
    filename = f"attendance_report_{att_date}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


# ---------------------------------------------------------------------
# 9. Real Alerts & Evidence Package Endpoints
# ---------------------------------------------------------------------
@api_bp.route("/alerts", methods=["GET"])
def get_alerts():
    import app
    severity = request.args.get("severity")
    alert_type = request.args.get("type")
    limit = int(request.args.get("limit", 50))

    conn = app.get_db()
    # Union across activeness_alerts and classroom_alerts
    query = """
        SELECT id, session_id, student_id, student_name, usn, alert_type, severity, detail, 
               photo_path, video_path, review_status, created_at, 'classroom' as source
        FROM activeness_alerts
        WHERE 1=1
    """
    params = []
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if alert_type:
        query += " AND alert_type = ?"
        params.append(alert_type)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    alerts = []
    for r in rows:
        item = dict(r)
        # Ensure photo URL starts with /activeness_photos/ or /static/
        if item.get("photo_path"):
            rel = os.path.basename(os.path.dirname(item["photo_path"]))
            fname = os.path.basename(item["photo_path"])
            item["photo_url"] = f"/activeness_photos/{item['session_id']}/{rel}/{fname}" if rel.startswith("event_") else f"/activeness_photos/{item['session_id']}/{fname}"
        else:
            item["photo_url"] = None

        if item.get("video_path"):
            rel = os.path.basename(os.path.dirname(item["video_path"]))
            fname = os.path.basename(item["video_path"])
            item["video_url"] = f"/activeness_photos/{item['session_id']}/{rel}/{fname}" if rel.startswith("event_") else f"/activeness_photos/{item['session_id']}/{fname}"
        else:
            item["video_url"] = None

        alerts.append(item)

    return jsonify({"success": True, "alerts": alerts})


@api_bp.route("/alerts/<int:alert_id>/action", methods=["POST"])
@api_login_required
def take_alert_action(alert_id):
    """Confirm or dismiss alert with reviewer attribution."""
    import app
    data = request.get_json(silent=True) or {}
    action = data.get("action")  # "CONFIRMED" or "DISMISSED"
    notes = data.get("notes", "").strip()

    if action not in ("CONFIRMED", "DISMISSED"):
        return jsonify({"success": False, "error": "Action must be CONFIRMED or DISMISSED."}), 400

    now_iso = datetime.now().isoformat()
    user = getattr(request, "current_user", {"username": "admin"})

    conn = app.get_db()
    conn.execute("""
        UPDATE activeness_alerts
        SET review_status = ?, reviewed_by = ?, reviewed_at = ?, review_notes = ?
        WHERE id = ?
    """, (action, user["username"], now_iso, notes, alert_id))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": f"Alert marked as {action}."})


# ---------------------------------------------------------------------
# 10. Dashboard Analytics
# ---------------------------------------------------------------------
@api_bp.route("/analytics", methods=["GET"])
def get_analytics():
    import app
    conn = app.get_db()
    today_str = date.today().isoformat()

    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    total_teachers = conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
    total_classes = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]

    # Today's attendance
    today_att = conn.execute("""
        SELECT 
            SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) as present,
            SUM(CASE WHEN LOWER(status) = 'absent' THEN 1 ELSE 0 END) as absent
        FROM attendance
        WHERE att_date = ?
    """, (today_str,)).fetchone()

    present_today = today_att[0] or 0
    absent_today = today_att[1] or 0
    att_rate = round((present_today / max(1, present_today + absent_today)) * 100, 1)

    # Alerts breakdown by category
    alert_breakdown = conn.execute("""
        SELECT alert_type, COUNT(*) as count
        FROM activeness_alerts
        GROUP BY alert_type
        ORDER BY count DESC
    """).fetchall()

    # Past 7 days attendance trend
    past_dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    trend = []
    for d in past_dates:
        row = conn.execute("""
            SELECT 
                SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) as pres,
                COUNT(*) as tot
            FROM attendance WHERE att_date = ?
        """, (d,)).fetchone()
        tot = row[1] or 0
        pres = row[0] or 0
        rate = round((pres / max(1, tot)) * 100, 1) if tot > 0 else 0
        trend.append({"date": d, "attendance_rate": rate, "present": pres, "total": tot})

    conn.close()

    return jsonify({
        "success": True,
        "metrics": {
            "total_students": total_students,
            "total_teachers": total_teachers,
            "total_classes": total_classes,
            "present_today": present_today,
            "absent_today": absent_today,
            "attendance_rate": att_rate,
            "alert_breakdown": [dict(b) for b in alert_breakdown],
            "weekly_trend": trend,
        }
    })


# ---------------------------------------------------------------------
# 11. WebSocket Live Stream Engine
# ---------------------------------------------------------------------
def register_websocket(sock):
    """
    Registers the /ws/classroom WebSocket endpoint.
    Processes live video frames from the browser camera:
    - YOLOv8 Person and Object detection
    - SFace face recognition & ClassroomTracker tracking
    - Activity & Behavior classification (sleeping, phone, posture)
    - Real-time duplicate-safe attendance recording
    - Multi-stage profiler FPS metrics
    """
    @sock.route("/ws/classroom")
    def ws_classroom(ws):
        import app
        tracker_inst = None
        active_class_id = None
        profiler = {
            "det": deque(maxlen=20),
            "rec": deque(maxlen=20),
            "trk": deque(maxlen=20),
            "e2e": deque(maxlen=20),
            "last_time": time.time(),
        }

        while True:
            msg = ws.receive()
            if msg is None:
                break

            t_start = time.time()
            try:
                data = json.loads(msg)
            except Exception:
                continue

            msg_type = data.get("type", "frame")
            if msg_type != "frame":
                continue

            img_b64 = data.get("image")
            class_id = data.get("class_id")
            if not img_b64:
                continue

            frame = app.decode_base64_image(img_b64)
            if frame is None:
                continue

            cid = int(class_id) if class_id else None
            if cid != active_class_id or tracker_inst is None:
                active_class_id = cid
                tracker_inst = _CLASSROOM_TRACKERS.setdefault(cid or 0, tracker.ClassroomTracker())

            session_info = _ACTIVE_CLASSROOM_SESSIONS.get(cid) if cid else None

            # Stage 1: YOLO Person & Object Detection
            t0 = time.time()
            yolo_res = yolo_detector.detect_objects_yolo(frame)
            t_det = time.time() - t0
            profiler["det"].append(t_det)

            # Stage 2: YuNet Face Detection & SFace Recognition
            t0 = time.time()
            student_gallery = face_engine.load_gallery(is_teacher=False)
            teacher_gallery = face_engine.load_gallery(is_teacher=True)
            faces = face_engine.detect_faces(frame)

            conn = app.get_db()
            class_students = {}
            if cid:
                class_students = {
                    r["id"]: r for r in conn.execute("SELECT * FROM students WHERE class_id = ?", (cid,)).fetchall()
                }

            # Run ClassroomTracker
            tracked_persons = tracker_inst.process_frame(frame, gallery=student_gallery, roster=class_students, timestamp=time.time())
            t_rec = time.time() - t0
            profiler["rec"].append(t_rec)

            # Stage 3: Live Behavior, Attendance Marking & Alerts
            new_alerts = []
            tracked_payload = []
            today_str = datetime.now().date().isoformat()
            now_iso = datetime.now().isoformat()

            for p in tracked_persons:
                bx, by, bw, bh = p["box"]
                status = p["status"]
                ident_src = p["identity_source"]
                matched_id = p["matched_id"]
                st_info = class_students.get(matched_id) if matched_id else None

                disp_name = st_info["name"] if st_info else ("Unknown" if status == "UNKNOWN" else f"Person #{p['track_id']}")
                disp_usn = st_info["usn"] if st_info else ""

                # Evaluate activity (e.g. Phone proximity from YOLO)
                activity = p.get("activity", "ATTENTIVE")
                has_phone = False
                for ph in yolo_res["phones"]:
                    px, py, pw, ph_h = ph["box"]
                    # If phone box overlaps or is near person
                    if abs(px - bx) < bw * 1.5 and abs(py - by) < bh * 1.5:
                        activity = "PHONE_USAGE"
                        has_phone = True
                        break

                # Prevent duplicate attendance: Check & mark attendance if active session
                is_marked = False
                if session_info and matched_id and status == "FACE_VERIFIED":
                    present_set = session_info.setdefault("present_students", set())
                    if matched_id not in present_set:
                        # Database idempotency check (Rule 17)
                        p_sid = session_info.get("period_session_id")
                        if p_sid:
                            existing = conn.execute(
                                "SELECT id FROM period_attendance WHERE session_id = ? AND student_id = ?",
                                (p_sid, matched_id)
                            ).fetchone()
                            if not existing:
                                conn.execute(
                                    "INSERT INTO period_attendance (session_id, student_id, status, marked_at) VALUES (?, ?, 'Present', ?)",
                                    (p_sid, matched_id, now_iso)
                                )
                            # Also upsert main attendance table
                            main_existing = conn.execute(
                                "SELECT id FROM attendance WHERE student_id = ? AND class_id = ? AND att_date = ?",
                                (matched_id, cid, today_str)
                            ).fetchone()
                            if not main_existing:
                                conn.execute(
                                    "INSERT INTO attendance (student_id, class_id, att_date, status) VALUES (?, ?, ?, 'Present')",
                                    (matched_id, cid, today_str)
                                )
                            conn.commit()

                        present_set.add(matched_id)
                        is_marked = True
                    else:
                        is_marked = True

                tracked_payload.append({
                    "track_id": p["track_id"],
                    "box": [bx, by, bw, bh],
                    "name": disp_name,
                    "usn": disp_usn,
                    "status": status,
                    "identity_source": ident_src,
                    "confidence": round(p.get("confidence", 0.0), 2),
                    "activity": activity,
                    "attendance_marked": is_marked
                })

            # Check teacher presence if waiting
            if session_info and not session_info.get("teacher_confirmed"):
                for f in faces:
                    _, emb = face_engine.extract_embedding(frame, f)
                    if emb is not None:
                        t_match = face_engine.match_face_against_gallery(emb, teacher_gallery)
                        if t_match["status"] == "VERIFIED":
                            t_row = conn.execute("SELECT name FROM teachers WHERE id = ?", (t_match["matched_id"],)).fetchone()
                            session_info["teacher_confirmed"] = True
                            session_info["teacher_name"] = t_row["name"] if t_row else "Verified Teacher"
                            break

            # Compute Profiler & FPS
            t_e2e = time.time() - t_start
            profiler["e2e"].append(t_e2e)

            def _fps(window):
                return round(1.0 / (sum(window) / len(window)), 1) if window and sum(window) > 0 else 0.0

            total_stud = len(class_students) if class_students else 0
            present_c = len(session_info.get("present_students", set())) if session_info else 0

            response = {
                "type": "detections",
                "timestamp": now_iso,
                "fps": {
                    "e2e_fps": _fps(profiler["e2e"]),
                    "detection_fps": _fps(profiler["det"]),
                    "recognition_fps": _fps(profiler["rec"]),
                    "latency_ms": round(t_e2e * 1000, 1),
                },
                "counts": {
                    "total": total_stud,
                    "present": present_c,
                    "absent": max(0, total_stud - present_c),
                    "percentage": round((present_c / max(1, total_stud)) * 100, 1)
                },
                "tracked": tracked_payload,
                "objects": yolo_res["all_objects"],
                "teacher_confirmed": session_info.get("teacher_confirmed", True) if session_info else True,
                "teacher_name": session_info.get("teacher_name", "") if session_info else "",
            }

            conn.close()
            ws.send(json.dumps(response))


def init_api(app, sock=None):
    """Registers API blueprint, WebSocket handler, and React SPA onto Flask app."""
    from flask import send_from_directory
    app.register_blueprint(api_bp)
    if sock is not None:
        register_websocket(sock)

    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")

    @app.route("/app")
    @app.route("/app/")
    @app.route("/app/<path:path>")
    def serve_frontend_spa(path=None):
        if path and os.path.exists(os.path.join(dist_dir, path)):
            return send_from_directory(dist_dir, path)
        if os.path.exists(os.path.join(dist_dir, "index.html")):
            return send_from_directory(dist_dir, "index.html")
        return "React Frontend is building. Please refresh in a moment.", 200
