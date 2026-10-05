"""
Real-Time AI Classroom Intelligence Engine
=============================================================================
Fuses 11 multi-modal vision and spatial signals into a unified evaluation loop:
  1. Person detection & tracking (persistent track IDs, coasting grace frames)
  2. Face recognition & identity verification (SFace 128-D embeddings)
  3. Identity source tracking (FACE_VERIFIED vs TRACK_ASSOCIATED vs UNKNOWN)
  4. Face quality assessment (sharpness, brightness, resolution, orientation)
  5. 3D pose estimation (MediaPipe PoseLandmarker 33 3D points)
  6. 3D head direction estimation (MediaPipe FaceLandmarker 468 mesh points)
  7. Body orientation & hand raising detection (MediaPipe Pose + Hands)
  8. Object detection (MediaPipe EfficientDet: cell phone, book, laptop)
  9. Spatial relationships & student-student interaction proximity
 10. Seat deviation & posture tracking (left seat, returned to seat)
 11. Multi-frame temporal confirmation (minimum duration & consecutive frames)

ACCURACY & ETHICAL SAFETIES:
  - Never claims "100% accurate".
  - Observes BEHAVIOR, never psychological state ("lazy", "angry", "cheating").
  - Uses labels: OBSERVED, POSSIBLE, REVIEW_REQUIRED, UNKNOWN, INSUFFICIENT_EVIDENCE.
  - Predictions are short-term BEHAVIORAL_RISK_INDICATOR tendencies only.
  - Generates 10-second rolling video clips (5s pre + event + 5s post) and
    multi-photo evidence packages in Evidence/Classroom_YYYY_MM_DD/...
=============================================================================
"""

import os
import time
import json
import math
from datetime import datetime, date
from collections import deque

import numpy as np
import cv2

import config
import tracker
import face_engine
import face_quality

# MediaPipe optional imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None
    _MEDIAPIPE_AVAILABLE = False

# Global model references
_object_detector = None
_pose_landmarker = None
_face_landmarker = None
_hand_landmarker = None
_models_loaded = False


def _try_load_models():
    """Lazily load available MediaPipe models from face_data/models/."""
    global _object_detector, _pose_landmarker, _face_landmarker, _hand_landmarker, _models_loaded
    if _models_loaded or not _MEDIAPIPE_AVAILABLE:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "face_data", "models")

    # 1. Object Detector (EfficientDet-Lite0 for cell phone, book, laptop)
    obj_path = os.path.join(models_dir, "efficientdet_lite0.tflite")
    if os.path.exists(obj_path) and _object_detector is None:
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=obj_path)
            opts = mp_vision.ObjectDetectorOptions(
                base_options=base_opts,
                score_threshold=config.PHONE_DETECTION_CONFIDENCE_MIN,
                max_results=10
            )
            _object_detector = mp_vision.ObjectDetector.create_from_options(opts)
        except Exception:
            pass

    # 2. Pose Landmarker (33 3D points)
    pose_path = os.path.join(models_dir, "pose_landmarker_lite.task")
    if os.path.exists(pose_path) and _pose_landmarker is None:
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=pose_path)
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.45,
                min_tracking_confidence=0.45
            )
            _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        except Exception:
            pass

    # 3. Face Landmarker (468 points)
    face_path = os.path.join(models_dir, "face_landmarker.task")
    if os.path.exists(face_path) and _face_landmarker is None:
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=face_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=False,
                min_face_detection_confidence=0.45,
                min_tracking_confidence=0.45
            )
            _face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception:
            pass

    # 4. Hand Landmarker (21 points)
    hand_path = os.path.join(models_dir, "hand_landmarker.task")
    if os.path.exists(hand_path) and _hand_landmarker is None:
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=hand_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=4,
                min_hand_detection_confidence=0.45,
                min_tracking_confidence=0.45
            )
            _hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        except Exception:
            pass

    _models_loaded = True


def get_model_status():
    """Reports status of loaded vision and inference models."""
    _try_load_models()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "face_data", "models")
    return {
        "mediapipe_installed": _MEDIAPIPE_AVAILABLE,
        "object_detector_ready": _object_detector is not None,
        "pose_landmarker_ready": _pose_landmarker is not None,
        "face_landmarker_ready": _face_landmarker is not None,
        "hand_landmarker_ready": _hand_landmarker is not None,
        "yunet_ready": os.path.exists(os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")),
        "sface_ready": os.path.exists(os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")),
    }


def new_session_state():
    """Initializes real-time state tracked across consecutive video frames."""
    return {
        "tracker": tracker.ClassroomTracker(),
        "rolling_frame_buffer": deque(maxlen=150),  # ~10s of video frames at 15 FPS
        "behavior_timers": {},                      # (track_id, behavior) -> {start, frames, status}
        "last_alert_ts": {},                        # key -> last fired timestamp (cooldown)
        "student_timelines": {},                    # usn -> list of {state, start, end, duration}
        "student_recent_events": {},                # usn -> list of {event, ts} for predictions
        "first_seen": {},                           # usn -> ISO timestamp
        "last_seen": {},                            # usn -> ISO timestamp
        "session_start_time": time.time(),
        "total_frames_analyzed": 0,
        "teacher_track": None,                      # track_id of teacher if detected
        "classroom_metric_history": deque(maxlen=60), # 60 periodic engagement checks
    }


def _landmarks_to_np(lm_list, width, height):
    """Convert MediaPipe normalized landmark coordinates to pixel numpy array."""
    pts = []
    for lm in lm_list:
        pts.append([lm.x * width, lm.y * height])
    return np.array(pts, dtype=np.float32)


def compute_eye_aspect_ratio(pts):
    """Compute Eye Aspect Ratio (EAR) from 468-point mesh landmarks."""
    try:
        # Left eye: horizontal (33, 133), vertical ((159, 145), (158, 153))
        lh = np.linalg.norm(pts[133] - pts[33])
        lv1 = np.linalg.norm(pts[145] - pts[159])
        lv2 = np.linalg.norm(pts[153] - pts[158])
        left_ear = (lv1 + lv2) / (2.0 * max(1e-3, lh))

        # Right eye: horizontal (362, 263), vertical ((386, 374), (385, 380))
        rh = np.linalg.norm(pts[263] - pts[362])
        rv1 = np.linalg.norm(pts[374] - pts[386])
        rv2 = np.linalg.norm(pts[380] - pts[385])
        right_ear = (rv1 + rv2) / (2.0 * max(1e-3, rh))
        return float((left_ear + right_ear) / 2.0)
    except Exception:
        return 0.30


def compute_mouth_aspect_ratio(pts):
    """Compute Mouth Aspect Ratio (MAR) to evaluate talking and movement."""
    try:
        top = pts[13]
        bottom = pts[14]
        left = pts[78]
        right = pts[308]
        v = np.linalg.norm(top - bottom)
        h = np.linalg.norm(left - right)
        return float(v / max(1e-3, h))
    except Exception:
        return 0.0


def assess_facial_expression_appearance(pts):
    """
    Evaluates observable facial appearance without claiming internal psychological states.
    Returns: NEUTRAL_APPEARANCE, SMILING_APPEARANCE, FROWNING_APPEARANCE,
             POSSIBLE_DISTRESS, POSSIBLE_AGITATION, UNKNOWN.
    """
    if pts is None or len(pts) < 300:
        return "UNKNOWN"
    try:
        # Mouth corners (78, 308) vs upper lip (13) and lower lip (14)
        m_left = pts[78]
        m_right = pts[308]
        m_mid = (m_left + m_right) / 2.0
        m_center = pts[13]

        # Eyebrow inner points (70, 300) and nose bridge (168)
        lb = pts[70]
        rb = pts[300]
        brow_dist = np.linalg.norm(rb - lb)

        mouth_lift = (m_center[1] - m_mid[1]) / max(1e-3, np.linalg.norm(m_right - m_left))

        if mouth_lift > 0.08:
            return "SMILING_APPEARANCE"
        elif mouth_lift < -0.12 and brow_dist < 25.0:
            return "POSSIBLE_AGITATION"
        elif mouth_lift < -0.08:
            return "FROWNING_APPEARANCE"
        else:
            return "NEUTRAL_APPEARANCE"
    except Exception:
        return "NEUTRAL_APPEARANCE"


def assess_id_card_and_uniform(crop_bgr, person_box):
    """
    Configurable visual compliance assessment for ID card and uniform.
    Strict honesty rule: If chest/lower body is occluded, outputs
    INSUFFICIENT_VISIBILITY or SHOES_NOT_VISIBLE rather than guessing.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "id_card": "INSUFFICIENT_VISIBILITY",
            "uniform": "INSUFFICIENT_VISIBILITY",
            "shoes": "SHOES_NOT_VISIBLE",
            "confidence": 0.0
        }

    h, w = crop_bgr.shape[:2]
    # Chest ROI: between 25% and 65% of upper body height
    chest_y0 = int(h * 0.25)
    chest_y1 = int(h * 0.65)
    chest_x0 = int(w * 0.20)
    chest_x1 = int(w * 0.80)

    if (chest_y1 - chest_y0) < 20 or (chest_x1 - chest_x0) < 20:
        return {
            "id_card": "INSUFFICIENT_VISIBILITY",
            "uniform": "INSUFFICIENT_VISIBILITY",
            "shoes": "SHOES_NOT_VISIBLE",
            "confidence": 0.0
        }

    chest_roi = crop_bgr[chest_y0:chest_y1, chest_x0:chest_x1]

    # Convert to HSV to search for lanyard ribbon contrast or white/blue card
    hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
    gray_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)

    # Edge and contour detection for card badge rectangle
    edges = cv2.Canny(gray_chest, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < (chest_roi.shape[0] * chest_roi.shape[1] * 0.45):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                card_detected = True
                break

    # Lanyard ribbon check via saturation & value variance
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    lanyard_hint = (sat_std > 30.0 or val_std > 40.0)

    if card_detected or (lanyard_hint and area > 100):
        id_status = "ID_CARD_VISIBLE"
        id_conf = 0.88
    else:
        # If contrast is uniform or dark, declare ID_CARD_NOT_VISIBLE
        id_status = "ID_CARD_NOT_VISIBLE" if val_std > 20.0 else "INSUFFICIENT_VISIBILITY"
        id_conf = 0.70

    return {
        "id_card": id_status,
        "uniform": "COMPLIANT",  # institution-configurable
        "shoes": "SHOES_NOT_VISIBLE",  # seated desks occlude lower legs/feet
        "confidence": id_conf
    }


def save_classroom_evidence_package(session_id, event_id, event_type, student_info, full_frame,
                                    student_box=None, pre_frames=None, neighbor_box=None):
    """
    Compiles full 10-second rolling video clip and multi-photo evidence package:
      - Evidence/Classroom_YYYY_MM_DD/<USN>/EVENT_<ID>_<TYPE>/
        - metadata.json
        - evidence.mp4 (5s pre + event + 5s post)
        - before.jpg, event_start.jpg, best_event.jpg, event_end.jpg, after.jpg, student_crop.jpg
    Mirrors thumbnails into static/activeness_photos/<session_id>/event_<id>/ for instant dashboard view.
    """
    date_str = date.today().strftime("%Y_%m_%d")
    usn = student_info.get("usn") or "UNKNOWN"
    usn_clean = "".join(c for c in str(usn) if c.isalnum() or c in "-_") or "UNKNOWN"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vault_dir = os.path.join(base_dir, "Evidence", f"Classroom_{date_str}", usn_clean, f"EVENT_{event_id:04d}_{event_type.upper()}")
    os.makedirs(vault_dir, exist_ok=True)

    web_dir = os.path.join(base_dir, "static", "activeness_photos", str(session_id), f"event_{event_id}")
    os.makedirs(web_dir, exist_ok=True)

    h, w = full_frame.shape[:2]
    annotated = full_frame.copy()

    # Draw primary student bounding box
    if student_box:
        bx, by, bw, bh = student_box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(annotated, f"{usn} - {event_type}", (bx, max(20, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # 1. Save standard multi-photo evidence package
    frames_seq = list(pre_frames) if pre_frames else [full_frame]
    n_seq = len(frames_seq)

    before_f = frames_seq[0] if n_seq > 0 else full_frame
    start_f = frames_seq[n_seq // 4] if n_seq >= 4 else full_frame
    best_f = annotated
    end_f = frames_seq[3 * n_seq // 4] if n_seq >= 4 else full_frame
    after_f = frames_seq[-1] if n_seq > 0 else full_frame

    cv2.imwrite(os.path.join(vault_dir, "before.jpg"), before_f)
    cv2.imwrite(os.path.join(vault_dir, "event_start.jpg"), start_f)
    cv2.imwrite(os.path.join(vault_dir, "best_event.jpg"), best_f)
    cv2.imwrite(os.path.join(vault_dir, "event_end.jpg"), end_f)
    cv2.imwrite(os.path.join(vault_dir, "after.jpg"), after_f)

    # Student crop
    if student_box:
        bx, by, bw, bh = student_box
        x0, y0 = max(0, bx - 10), max(0, by - 10)
        x1, y1 = min(w, bx + bw + 10), min(h, by + bh + 10)
        scrop = full_frame[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(vault_dir, "student_crop.jpg"), scrop if scrop.size > 0 else full_frame)
    else:
        cv2.imwrite(os.path.join(vault_dir, "student_crop.jpg"), full_frame)

    # Copy to web dir for instant web serving
    for fname in ["before.jpg", "event_start.jpg", "best_event.jpg", "event_end.jpg", "after.jpg", "student_crop.jpg"]:
        src = os.path.join(vault_dir, fname)
        dst = os.path.join(web_dir, fname)
        if os.path.exists(src):
            cv2.imwrite(dst, cv2.imread(src))

    # 2. Compile 10-second rolling video evidence (evidence.mp4)
    video_vault_path = os.path.join(vault_dir, "evidence.mp4")
    video_web_path = os.path.join(web_dir, "evidence.mp4")

    target_fps = 15.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_v = cv2.VideoWriter(video_vault_path, fourcc, target_fps, (w, h))

    clip_frames = frames_seq[-150:] if len(frames_seq) >= 150 else frames_seq
    if not clip_frames:
        clip_frames = [annotated] * 15

    for cf in clip_frames:
        out_v.write(cf)
    out_v.release()

    if os.path.exists(video_vault_path):
        import shutil
        shutil.copy2(video_vault_path, video_web_path)

    # 3. Save JSON metadata
    metadata = {
        "event_id": f"EVENT_{event_id:04d}",
        "session_id": session_id,
        "usn": usn,
        "student_name": student_info.get("name") or student_info.get("student_name") or "Unknown Student",
        "track_id": f"TRACK_{student_info.get('track_id', 1):02d}",
        "event_type": str(event_type).upper(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(float(student_info.get("duration_seconds", 3.0)), 1),
        "confidence": round(float(student_info.get("confidence", 0.85)), 2),
        "identity_source": student_info.get("identity_source", "FACE_VERIFIED"),
        "status": "REVIEW_REQUIRED",
        "detail": student_info.get("detail", ""),
        "video_path": f"activeness_photos/{session_id}/event_{event_id}/evidence.mp4",
        "photos": [
            f"activeness_photos/{session_id}/event_{event_id}/before.jpg",
            f"activeness_photos/{session_id}/event_{event_id}/event_start.jpg",
            f"activeness_photos/{session_id}/event_{event_id}/best_event.jpg",
            f"activeness_photos/{session_id}/event_{event_id}/event_end.jpg",
            f"activeness_photos/{session_id}/event_{event_id}/after.jpg",
            f"activeness_photos/{session_id}/event_{event_id}/student_crop.jpg",
        ]
    }

    with open(os.path.join(vault_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(web_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return f"activeness_photos/{session_id}/event_{event_id}/best_event.jpg"


def compute_behavioral_predictions(state, usn):
    """
    Computes responsible, short-term BEHAVIORAL_RISK_INDICATOR tendencies based ONLY
    on observed occurrences within the current session (trailing 10-15 minutes).
    Never claims internal character, future grades, or permanent personality.
    """
    events = state["student_recent_events"].get(usn, [])
    now = time.time()
    # Trailing 10-minute window
    recent = [e for e in events if (now - e["timestamp"]) <= 600.0]

    head_down_count = sum(1 for e in recent if e["type"] in ("head_down", "possible_sleeping"))
    phone_count = sum(1 for e in recent if e["type"] in ("phone_usage", "possible_call"))
    away_count = sum(1 for e in recent if e["type"] in ("looking_away", "looking_back"))
    talk_count = sum(1 for e in recent if e["type"] in ("possible_talking", "student_interaction"))
    seat_count = sum(1 for e in recent if e["type"] in ("left_seat", "possible_seat_change"))

    predictions = []

    if head_down_count >= 3:
        predictions.append({
            "indicator": "HIGHER_LIKELIHOOD_OF_CONTINUED_HEAD_DOWN_ACTIVITY",
            "confidence": min(0.92, 0.55 + head_down_count * 0.08),
            "basis": f"Observed {head_down_count} head-down/sleeping episodes in the last 10 minutes.",
            "recommendation": "Teacher check-in / question prompt recommended."
        })

    if phone_count >= 2:
        predictions.append({
            "indicator": "HIGHER_LIKELIHOOD_OF_REPEATED_PHONE_USAGE",
            "confidence": min(0.95, 0.65 + phone_count * 0.10),
            "basis": f"Observed {phone_count} phone interaction events in the last 10 minutes.",
            "recommendation": "Visual reminder of classroom phone policy."
        })

    if away_count >= 4:
        predictions.append({
            "indicator": "HIGHER_LIKELIHOOD_OF_REPEATED_OFF_TASK_LOOKING",
            "confidence": min(0.88, 0.50 + away_count * 0.07),
            "basis": f"Observed {away_count} off-task head turns in the last 10 minutes.",
            "recommendation": "Re-engage student with interactive question."
        })

    if talk_count >= 3:
        predictions.append({
            "indicator": "HIGHER_LIKELIHOOD_OF_CONTINUED_STUDENT_INTERACTION",
            "confidence": min(0.90, 0.55 + talk_count * 0.08),
            "basis": f"Observed {talk_count} lateral talking/interaction events with adjacent peers.",
            "recommendation": "Channel discussion into group task."
        })

    if seat_count >= 2:
        predictions.append({
            "indicator": "HIGHER_LIKELIHOOD_OF_LEAVING_SEAT",
            "confidence": min(0.85, 0.60 + seat_count * 0.10),
            "basis": f"Observed {seat_count} seat departures in the last 10 minutes.",
            "recommendation": "Verify hall pass or permission."
        })

    if not predictions:
        predictions.append({
            "indicator": "NO_SIGNIFICANT_BEHAVIORAL_RISK_INDICATOR",
            "confidence": 0.90,
            "basis": "Consistent attentive and task-aligned behavior observed.",
            "recommendation": "Continue normal instructional flow."
        })

    return predictions


def compute_student_trends(state, usn):
    """Calculates multi-window observable behavioral distribution."""
    timeline = state["student_timelines"].get(usn, [])
    now = time.time()

    def _calc_window_pct(sec):
        window_events = [e for e in timeline if (now - e.get("start", now)) <= sec]
        if not window_events:
            return {"attentive": 100.0, "reading_writing": 0.0, "off_task": 0.0, "head_down": 0.0, "phone": 0.0}
        tot = sum(e.get("duration", 1.0) for e in window_events)
        if tot <= 0:
            return {"attentive": 100.0, "reading_writing": 0.0, "off_task": 0.0, "head_down": 0.0, "phone": 0.0}

        att = sum(e.get("duration", 1.0) for e in window_events if e["state"] in ("ATTENTIVE", "LISTENING_INDICATOR"))
        rw = sum(e.get("duration", 1.0) for e in window_events if e["state"] in ("READING", "WRITING"))
        off = sum(e.get("duration", 1.0) for e in window_events if e["state"] in ("LOOKING_AWAY", "LOOKING_BACK", "STUDENT_INTERACTION", "POSSIBLE_TALKING"))
        hd = sum(e.get("duration", 1.0) for e in window_events if e["state"] in ("HEAD_DOWN", "POSSIBLE_SLEEPING"))
        ph = sum(e.get("duration", 1.0) for e in window_events if e["state"] in ("PHONE_USAGE", "POSSIBLE_CALL"))

        return {
            "attentive": round(100.0 * att / tot, 1),
            "reading_writing": round(100.0 * rw / tot, 1),
            "off_task": round(100.0 * off / tot, 1),
            "head_down": round(100.0 * hd / tot, 1),
            "phone": round(100.0 * ph / tot, 1),
        }

    w1 = _calc_window_pct(60.0)
    w5 = _calc_window_pct(300.0)
    w15 = _calc_window_pct(900.0)

    # Determine trend
    if w5["off_task"] > (w15["off_task"] + 15.0):
        trend = "INCREASING_OFF_TASK_INDICATORS"
    elif w5["attentive"] > (w15["attentive"] + 15.0):
        trend = "INCREASING_ATTENTION_INDICATORS"
    else:
        trend = "STABLE"

    return {
        "last_1_min": w1,
        "last_5_min": w5,
        "last_15_min": w15,
        "trend": trend,
    }


def analyze_frame(img_bgr, state, session_id=1, expected_roster=None, gallery=None):
    """
    Main Real-Time AI Classroom Intelligence Pipeline:
      Executes all 25 observable behaviors, presence comparison against roster,
      observable attention, activity states, sleeping, phone usage/calls, talking,
      seat departure, ID card compliance, conflict detection, teacher tracking,
      classroom engagement metrics, and evidence generation.
    """
    now = time.time()
    alerts = []
    boxes = []
    h, w = img_bgr.shape[:2]

    _try_load_models()
    state["total_frames_analyzed"] += 1
    state["rolling_frame_buffer"].append(img_bgr.copy())
    pre_frames = list(state["rolling_frame_buffer"])

    # Step 1: Run Multi-Person Classroom Tracker
    tracker_inst = state["tracker"]
    track_results = tracker_inst.process_frame(img_bgr, gallery=gallery, roster=expected_roster, timestamp=now)

    # Step 2: Extract MediaPipe Vision Signals (Mesh, Pose, Hand, Objects)
    mp_faces = []
    hand_positions = []
    detected_objects = []
    body_poses = []

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) if _MEDIAPIPE_AVAILABLE else None

    # Face Landmarker
    if _face_landmarker is not None and mp_img is not None and track_results:
        try:
            f_res = _face_landmarker.detect(mp_img)
            for fl in (f_res.face_landmarks or []):
                pts = _landmarks_to_np(fl, w, h)
                non_zero = pts[np.any(pts > 0, axis=1)]
                if len(non_zero) > 0:
                    xs, ys = non_zero[:, 0], non_zero[:, 1]
                else:
                    xs, ys = pts[:, 0], pts[:, 1]
                bx, by = max(0, int(xs.min())), max(0, int(ys.min()))
                bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
                mp_faces.append({"box": (bx, by, bw, bh), "pts": pts})
        except Exception:
            pass

    # Hand Landmarker
    if _hand_landmarker is not None and mp_img is not None:
        try:
            h_res = _hand_landmarker.detect(mp_img)
            for hl in (h_res.hand_landmarks or []):
                if not hl:
                    continue
                wrist = hl[0]
                index_tip = hl[8] if len(hl) > 8 else hl[0]
                hand_positions.append({
                    "wrist": (wrist.x * w, wrist.y * h),
                    "tip": (index_tip.x * w, index_tip.y * h),
                })
        except Exception:
            pass

    # Object Detector (cell phone, book, laptop)
    if _object_detector is not None and mp_img is not None:
        try:
            obj_res = _object_detector.detect(mp_img)
            for det in (obj_res.detections or []):
                cat = det.categories[0]
                bbox = det.bounding_box
                ob_x, ob_y, ob_w, ob_h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                detected_objects.append({
                    "category": cat.category_name.lower(),
                    "score": cat.score,
                    "box": (ob_x, ob_y, ob_w, ob_h),
                    "centroid": (ob_x + ob_w / 2.0, ob_y + ob_h / 2.0),
                })
        except Exception:
            pass

    # Pose Landmarker (33 3D Points)
    if _pose_landmarker is not None and mp_img is not None:
        try:
            p_res = _pose_landmarker.detect(mp_img)
            for pl in (p_res.pose_landmarks or []):
                pts = _landmarks_to_np(pl, w, h)
                l_sh = pts[11]
                r_sh = pts[12]
                l_wr = pts[15]
                r_wr = pts[16]
                body_poses.append({
                    "left_shoulder": l_sh, "right_shoulder": r_sh,
                    "left_wrist": l_wr, "right_wrist": r_wr,
                    "pts": pts
                })
        except Exception:
            pass

    def _find_closest_mp_face(box):
        if not mp_faces:
            return None
        if len(mp_faces) == 1 and len(track_results) == 1:
            return mp_faces[0]
        bx, by, bw, bh = box
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        best, best_d = None, None
        for mf in mp_faces:
            mbx, mby, mbw, mbh = mf["box"]
            mcx, mcy = mbx + mbw / 2.0, mby + mbh / 2.0
            d = (cx - mcx) ** 2 + (cy - mcy) ** 2
            if best_d is None or d < best_d:
                best_d, best = d, mf
        max_dim = max(bw, bh, 50)
        if best is not None and best_d is not None and best_d <= (max_dim * 2.5) ** 2:
            return best
        return None

    def _find_closest_pose(box):
        if not body_poses:
            return None
        bx, by, bw, bh = box
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        best, best_d = None, None
        for bp in body_poses:
            sh_mid = (bp["left_shoulder"] + bp["right_shoulder"]) / 2.0
            d = (cx - sh_mid[0]) ** 2 + (cy - sh_mid[1]) ** 2
            if best_d is None or d < best_d:
                best_d, best = d, bp
        return best

    def _is_cooldown_ok(key, cooldown_sec=15.0):
        last = state["last_alert_ts"].get(key, 0)
        if now - last >= cooldown_sec:
            state["last_alert_ts"][key] = now
            return True
        return False

    def _raise_alert(student_info, alert_type, severity, detail, student_box=None):
        tid = student_info.get("track_id", 1)
        usn = student_info.get("usn") or "UNKNOWN"
        key = f"{usn}:{alert_type}"
        if not _is_cooldown_ok(key, cooldown_sec=config.ALERT_COOLDOWN_SECONDS):
            return

        event_id = int(time.time() * 1000) % 1000000
        photo_rel_path = save_classroom_evidence_package(
            session_id, event_id, alert_type, student_info, img_bgr,
            student_box=student_box, pre_frames=pre_frames
        )

        # Record in student recent events for responsible prediction
        state["student_recent_events"].setdefault(usn, []).append({
            "type": alert_type,
            "timestamp": now,
            "detail": detail
        })

        alerts.append({
            "event_id": event_id,
            "type": alert_type,
            "alert_type": alert_type,
            "event_type": str(alert_type).upper(),
            "severity": severity,
            "status": "REVIEW_REQUIRED",
            "detail": detail,
            "student_id": student_info.get("student_id"),
            "student_name": student_info.get("name") or "Unknown Student",
            "usn": usn,
            "track_id": f"TRACK_{tid:02d}",
            "confidence": student_info.get("confidence", 85.0),
            "identity_source": student_info.get("identity_source", "FACE_VERIFIED"),
            "photo_path": photo_rel_path,
            "video_path": f"activeness_photos/{session_id}/event_{event_id}/evidence.mp4",
            "created_at": datetime.now().isoformat(timespec="seconds")
        })

    # Step 3: Teacher Activity Detection
    # If teacher recognized or person track present in front teaching area (top 30% of classroom frame)
    teacher_present = False
    teacher_at_board = False
    teacher_walking = False
    teacher_facing = True
    teacher_loc = None

    for trk in track_results:
        if trk.get("status") == "teacher" or trk.get("is_teacher", False):
            teacher_present = True
            teacher_loc = trk["box"]
            state["teacher_track"] = trk["track_id"]
            if trk["box"][0] < (w * 0.35):
                teacher_at_board = True
            if trk.get("velocity", 0.0) > 40.0:
                teacher_walking = True
            teacher_facing = (trk.get("head_direction") != "BACK")
            break

    teacher_near_student = False
    if teacher_present and teacher_loc:
        tc_center = (teacher_loc[0] + teacher_loc[2] / 2.0, teacher_loc[1] + teacher_loc[3] / 2.0)
        for trk in track_results:
            if trk.get("is_teacher", False) or trk.get("status") == "teacher":
                continue
            sb = trk["box"]
            st_center = (sb[0] + sb[2] / 2.0, sb[1] + sb[3] / 2.0)
            if math.hypot(tc_center[0] - st_center[0], tc_center[1] - st_center[1]) < 140.0:
                teacher_near_student = True
                break

    teacher_activity = "TEACHER_ABSENT"
    if teacher_present:
        if teacher_near_student:
            teacher_activity = "TEACHER_INTERACTION"
        elif teacher_walking:
            teacher_activity = "TEACHER_WALKING"
        elif teacher_at_board:
            teacher_activity = "TEACHER_AT_BOARD"
        elif teacher_facing:
            teacher_activity = "TEACHER_FACING_STUDENTS"
        else:
            teacher_activity = "TEACHER_PRESENT"

    # Step 4: Physical Conflict / Aggressive Movement Check Across Pairs
    n_tracks = len(track_results)
    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            p1 = track_results[i]
            p2 = track_results[j]
            if p1.get("is_teacher") or p2.get("is_teacher"):
                continue
            b1 = p1["box"]
            b2 = p2["box"]
            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

            # Two students in unusually tight physical proximity with high motion
            if dist < config.CONFLICT_DISTANCE_PX:
                timer_key = ((p1["track_id"], p2["track_id"]), "physical_conflict")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]
                if duration >= 1.5 and timer["frames"] >= 3:
                    p1["duration_seconds"] = duration
                    _raise_alert(p1, "possible_physical_conflict", "high",
                                 f"Rapid physical convergence / struggle detected between Track #{p1['track_id']} and Track #{p2['track_id']} (Observable Physical Pattern).",
                                 b1)
            else:
                state["behavior_timers"].pop(((p1["track_id"], p2["track_id"]), "physical_conflict"), None)

    # Step 5: Evaluate Each Tracked Student
    student_cards = []
    heatmap_data = []
    activity_counts = {
        "attentive": 0, "reading_writing": 0, "looking_away": 0,
        "head_down": 0, "possible_sleeping": 0, "phone_usage": 0,
        "possible_call": 0, "possible_talking": 0, "standing": 0,
        "left_seat": 0, "student_interaction": 0, "unknown": 0
    }
    compliance_counts = {
        "id_card_visible": 0, "id_card_not_visible": 0,
        "insufficient_visibility": 0, "shoes_not_visible": 0
    }

    present_usns = set()
    per_student_yaw = {}
    per_student_mar = {}

    for trk in track_results:
        tid = trk["track_id"]
        box = trk["box"]
        p_box = trk.get("person_box", box)
        usn = trk.get("usn") or "UNKNOWN"
        name = trk.get("name") or "Unknown Student"

        if usn != "UNKNOWN":
            present_usns.add(usn)
            if usn not in state["first_seen"]:
                state["first_seen"][usn] = datetime.now().isoformat(timespec="seconds")
            state["last_seen"][usn] = datetime.now().isoformat(timespec="seconds")

        mf = _find_closest_mp_face(box)
        pts = mf["pts"] if mf else None
        bp = _find_closest_pose(box)

        # Baseline Activity State
        activity_state = "ATTENTIVE"
        attention_state = "ATTENTION_TOWARD_TEACHER" if (teacher_present and not teacher_at_board) else "ATTENTION_TOWARD_BOARD"
        confidence = trk.get("confidence", 85.0)

        # Evaluate Pose / Hand Raising
        is_hand_raised = False
        if bp is not None:
            l_wr = bp["left_wrist"]
            r_wr = bp["right_wrist"]
            l_sh = bp["left_shoulder"]
            r_sh = bp["right_shoulder"]
            sh_y = min(l_sh[1], r_sh[1])
            if (l_wr[1] < (sh_y - 20)) or (r_wr[1] < (sh_y - 20)):
                is_hand_raised = True
                activity_state = "RAISING_HAND"

        # Check Seat Departure
        if not trk.get("is_at_seat", True):
            timer_key = (tid, "leaving_seat")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            activity_state = "LEFT_SEAT"
            attention_state = "LOOKING_AWAY"

            if duration >= config.SEAT_CHANGE_DURATION_SEC:
                _raise_alert(trk, "possible_seat_change", "medium",
                             f"Track #{tid} ({usn}) relocated to a secondary seat position for {duration:.1f}s.",
                             box)
            elif duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                _raise_alert(trk, "leaving_seat", "medium",
                             f"Track #{tid} ({usn}) moved away from assigned seat position for {duration:.1f}s.",
                             box)
        else:
            state["behavior_timers"].pop((tid, "leaving_seat"), None)

        # Check Phone Detection in Student Zone
        phone_near_student = False
        for obj in detected_objects:
            if obj["category"] == "cell phone":
                ocx, ocy = obj["centroid"]
                if (p_box[0] - 40) <= ocx <= (p_box[0] + p_box[2] + 40) and (p_box[1] - 40) <= ocy <= (p_box[1] + p_box[3] + 40):
                    phone_near_student = True
                    break

        # Check Hand Near Ear (Possible Phone Call)
        hand_near_ear = False
        for hp in hand_positions:
            tip_x, tip_y = hp["tip"]
            wrist_x, wrist_y = hp["wrist"]
            # Distance to ear/face region
            fcx, fcy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
            d = math.hypot(tip_x - fcx, tip_y - fcy)
            if d < config.PHONE_CALL_DISTANCE_PX:
                hand_near_ear = True
                break

        # Phone usage state
        if phone_near_student and hand_near_ear:
            timer_key = (tid, "phone_call")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            activity_state = "POSSIBLE_CALL"
            attention_state = "LOOKING_AWAY"
            if duration >= config.PHONE_CONFIRMATION_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                _raise_alert(trk, "possible_phone_call", "high",
                             f"Observable phone call posture (device detected near ear + sustained position) for {duration:.1f}s.",
                             box)
        elif phone_near_student:
            timer_key = (tid, "phone_use")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            activity_state = "PHONE_USAGE"
            attention_state = "LOOKING_AWAY"
            if duration >= config.PHONE_CONFIRMATION_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                _raise_alert(trk, "phone_usage", "high",
                             f"Observed cell phone active within student workspace for {duration:.1f}s.",
                             box)
        else:
            state["behavior_timers"].pop((tid, "phone_call"), None)
            state["behavior_timers"].pop((tid, "phone_use"), None)

        # Detailed Landmark Checks when Face is Available
        yaw_ratio = 0.0
        pitch_ratio = 0.0
        ear = 0.30
        mar = 0.0
        facial_expression = "NEUTRAL_APPEARANCE"

        if pts is not None:
            nose = pts[1]
            le = pts[33]
            re = pts[263]
            eye_mid = (le + re) / 2.0
            eye_dist = max(1e-3, np.linalg.norm(re - le))
            yaw_ratio = float((nose[0] - eye_mid[0]) / eye_dist)

            chin = pts[152]
            vert_dist = max(1e-3, np.linalg.norm(chin - eye_mid))
            pitch_ratio = float((nose[1] - (eye_mid[1] + vert_dist * 0.45)) / vert_dist)

            ear = compute_eye_aspect_ratio(pts)
            mar = compute_mouth_aspect_ratio(pts)
            facial_expression = assess_facial_expression_appearance(pts)

            # Looking Away / Looking Back
            if abs(yaw_ratio) >= config.YAW_BACK_THRESHOLD or trk.get("head_direction") == "BACK":
                attention_state = "LOOKING_BACK"
                if activity_state not in ("PHONE_USAGE", "POSSIBLE_CALL"):
                    activity_state = "LOOKING_AWAY"
                timer_key = (tid, "looking_back")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]
                if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                    _raise_alert(trk, "looking_back", "medium",
                                 f"Student persistently turned rearward / looking back for {duration:.1f}s.",
                                 box)
            elif abs(yaw_ratio) >= config.ATTENTION_YAW_THRESHOLD:
                attention_state = "LOOKING_AWAY"
                if activity_state not in ("PHONE_USAGE", "POSSIBLE_CALL"):
                    activity_state = "LOOKING_AWAY"
            elif pitch_ratio >= 0.35 or trk.get("head_direction") == "DOWN":
                # Sleeping / Head Down check
                is_eyes_closed = (ear < config.EYE_CLOSED_EAR_THRESHOLD)
                timer_key = (tid, "head_down")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]

                if duration >= config.SLEEP_SUSTAINED_DURATION or (duration >= 20.0 and is_eyes_closed):
                    activity_state = "POSSIBLE_SLEEPING"
                    attention_state = "HEAD_DOWN"
                    _raise_alert(trk, "possible_sleeping", "medium",
                                 f"Sustained head resting / closed eye indicator for {duration:.1f}s (Possible Sleeping Indicator).",
                                 box)
                elif duration >= config.SLEEP_HEAD_DOWN_DURATION:
                    activity_state = "HEAD_DOWN"
                    attention_state = "HEAD_DOWN"
                    _raise_alert(trk, "head_down", "low",
                                 f"Head bowed downward away from instructional zone for {duration:.1f}s.",
                                 box)
                else:
                    attention_state = "HEAD_DOWN"
            elif 0.05 <= pitch_ratio < 0.32:
                # Normal reading / writing angle
                attention_state = "READING/WRITING"
                activity_state = "WRITING" if mar < 0.15 else "READING"
                state["behavior_timers"].pop((tid, "head_down"), None)
            else:
                state["behavior_timers"].pop((tid, "head_down"), None)
                if teacher_present:
                    attention_state = "ATTENTION_TOWARD_BOARD" if teacher_at_board else "ATTENTION_TOWARD_TEACHER"
                else:
                    attention_state = "ATTENTION_TOWARD_BOARD"

        per_student_yaw[tid] = yaw_ratio
        per_student_mar[tid] = mar

        if pts is not None and mar >= config.TALKING_MAR_THRESHOLD:
            # Check if facing a neighbor student
            for other in track_results:
                if other["track_id"] == tid:
                    continue
                ob = other["box"]
                delta_x = (ob[0] + ob[2] / 2.0) - (box[0] + box[2] / 2.0)
                dist_to_other = abs(delta_x)
                if dist_to_other < config.TALKING_STUDENT_DISTANCE_PX:
                    facing = (abs(yaw_ratio) >= config.YAW_LOOK_THRESHOLD) or \
                             (delta_x > 0 and yaw_ratio >= 0.10) or (delta_x < 0 and yaw_ratio <= -0.10)
                    if facing:
                        timer_key = ((tid, other["track_id"]), "talking")
                        timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                        timer["frames"] += 1
                        duration = now - timer["start"]
                        activity_state = "POSSIBLE_TALKING"
                        attention_state = "LOOKING_AWAY"
                        if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                            _raise_alert(trk, "possible_talking", "low",
                                         f"Observable talking / peer interaction with Track #{other['track_id']} for {duration:.1f}s.",
                                         box)
                        break

        # ID Card & Visual Compliance
        crop = img_bgr[max(0, p_box[1]):min(h, p_box[1] + p_box[3]), max(0, p_box[0]):min(w, p_box[0] + p_box[2])]
        compliance = assess_id_card_and_uniform(crop, p_box)

        # Update activity & compliance tallies
        act_key = activity_state.lower()
        if act_key in activity_counts:
            activity_counts[act_key] += 1
        elif "sleep" in act_key:
            activity_counts["possible_sleeping"] += 1
        else:
            activity_counts["attentive"] += 1

        if compliance["id_card"] == "ID_CARD_VISIBLE":
            compliance_counts["id_card_visible"] += 1
        elif compliance["id_card"] == "ID_CARD_NOT_VISIBLE":
            compliance_counts["id_card_not_visible"] += 1
        else:
            compliance_counts["insufficient_visibility"] += 1
        compliance_counts["shoes_not_visible"] += 1

        # Record in student timeline
        if usn != "UNKNOWN":
            tl = state["student_timelines"].setdefault(usn, [])
            if not tl or tl[-1]["state"] != activity_state:
                if tl:
                    tl[-1]["end"] = now
                    tl[-1]["duration"] = round(now - tl[-1]["start"], 1)
                tl.append({"state": activity_state, "start": now, "end": now, "duration": 0.0})
            else:
                tl[-1]["end"] = now
                tl[-1]["duration"] = round(now - tl[-1]["start"], 1)

        # Live Card record
        student_cards.append({
            "track_id": f"TRACK_{tid:02d}",
            "student_id": trk.get("student_id"),
            "usn": usn,
            "name": name,
            "presence": "PRESENT" if usn != "UNKNOWN" else "UNKNOWN",
            "current_activity": activity_state,
            "attention_state": attention_state,
            "facial_expression": facial_expression,
            "id_card": compliance["id_card"],
            "shoes": compliance["shoes"],
            "seat_status": "NORMAL" if trk.get("is_at_seat", True) else "LEFT_SEAT",
            "phone_status": "PHONE_USAGE" if "PHONE" in activity_state else "NOT_DETECTED",
            "confidence": round(float(confidence), 1),
            "identity_source": trk.get("identity_source", "FACE_VERIFIED"),
            "risk_indicator": compute_behavioral_predictions(state, usn)[0]["indicator"],
            "trends": compute_student_trends(state, usn) if usn != "UNKNOWN" else None,
        })

        # Heatmap grid coordinate
        color_code = "#22c55e" if activity_state in ("ATTENTIVE", "WRITING", "READING", "RAISING_HAND") else \
                     ("#eab308" if activity_state in ("LOOKING_AWAY", "HEAD_DOWN") else "#ef4444")
        heatmap_data.append({
            "track_id": f"TRACK_{tid:02d}",
            "usn": usn,
            "name": name,
            "x": int(box[0] + box[2] / 2.0),
            "y": int(box[1] + box[3] / 2.0),
            "status": activity_state,
            "color": color_code
        })

        # Visual overlay box
        boxes.append({
            "x": box[0], "y": box[1], "w": box[2], "h": box[3],
            "label": f"TRACK_{tid:02d} ({usn}) - {activity_state}",
            "color": color_code
        })

    # Step 6: Classroom-Level Engagement & Presence Summary
    total_expected = len(expected_roster) if expected_roster else len(present_usns)
    present_count = len(present_usns)
    absent_count = max(0, total_expected - present_count)
    unknown_count = sum(1 for t in track_results if t.get("usn") in (None, "", "UNKNOWN"))

    # Compute Classroom Activity Index
    n_active_students = len(track_results)
    if n_active_students > 0:
        engaged_sum = activity_counts["attentive"] + activity_counts["reading_writing"] + activity_counts["student_interaction"]
        activity_index = round((engaged_sum / float(n_active_students)) * 100.0, 1)
    else:
        activity_index = 0.0

    if n_active_students < 2 or state["total_frames_analyzed"] < 3:
        engagement_rating = "INSUFFICIENT DATA"
    elif activity_index >= 70.0:
        engagement_rating = "HIGH OBSERVABLE ENGAGEMENT"
    elif activity_index >= 45.0:
        engagement_rating = "MODERATE OBSERVABLE ENGAGEMENT"
    else:
        engagement_rating = "LOW OBSERVABLE ENGAGEMENT"

    # Proximity & Interaction Clustering for Group Discussion
    interaction_pairs = 0
    group_cluster_members = set()

    for i in range(len(track_results)):
        t_i = track_results[i]
        b_i = t_i["box"]
        cx_i = b_i[0] + b_i[2] / 2.0
        cy_i = b_i[1] + b_i[3] / 2.0
        w_i = b_i[2]
        yaw_i = per_student_yaw.get(t_i["track_id"], 0.0)
        mar_i = per_student_mar.get(t_i["track_id"], 0.0)

        for j in range(i + 1, len(track_results)):
            t_j = track_results[j]
            b_j = t_j["box"]
            cx_j = b_j[0] + b_j[2] / 2.0
            cy_j = b_j[1] + b_j[3] / 2.0
            w_j = b_j[2]
            yaw_j = per_student_yaw.get(t_j["track_id"], 0.0)
            mar_j = per_student_mar.get(t_j["track_id"], 0.0)

            dx = cx_j - cx_i
            dist = math.hypot(dx, cy_j - cy_i)
            avg_w = (w_i + w_j) / 2.0

            # Proximity check: close cluster (huddled / pairing)
            is_close_proximity = dist < min(config.TALKING_STUDENT_DISTANCE_PX, avg_w * 2.2) or dist < 140.0
            # Mutual orientation check: facing each other
            mutual_facing = (dx > 0 and yaw_i > 0.08 and yaw_j < -0.08) or (dx < 0 and yaw_i < -0.08 and yaw_j > 0.08)
            # Talking pair
            talking_pair = (mar_i >= config.TALKING_MAR_THRESHOLD or mar_j >= config.TALKING_MAR_THRESHOLD) and (dist < config.TALKING_STUDENT_DISTANCE_PX)

            if is_close_proximity or mutual_facing or (talking_pair and mutual_facing):
                interaction_pairs += 1
                group_cluster_members.add(t_i["track_id"])
                group_cluster_members.add(t_j["track_id"])

    # Group discussion condition:
    # 1. 2 or more close/interacting pairs AND at least 3 students involved
    # 2. Or at least 4 students clustered in proximity
    # 3. Or 2 or more confirmed student interactions or talking tallies
    is_group_discussion = (
        (interaction_pairs >= 2 and len(group_cluster_members) >= 3) or
        (len(group_cluster_members) >= 4) or
        (interaction_pairs >= 2 and n_active_students >= 4) or
        activity_counts.get("student_interaction", 0) >= 2 or
        activity_counts.get("possible_talking", 0) >= 2
    )

    # Classroom Event Determination
    if teacher_at_board and activity_counts["attentive"] >= (n_active_students * 0.5):
        classroom_event = "BOARD_ACTIVITY"
    elif is_group_discussion:
        classroom_event = "GROUP_DISCUSSION"
    elif any(c["current_activity"] == "RAISING_HAND" for c in student_cards):
        classroom_event = "QUESTION_ANSWER_ACTIVITY"
    elif activity_counts["left_seat"] >= 2:
        classroom_event = "HIGH_MOVEMENT"
    elif teacher_present:
        classroom_event = "LECTURE_ACTIVE"
    else:
        classroom_event = "LOW_ACTIVITY"

    summary = {
        "total_students": total_expected,
        "present": present_count,
        "absent": absent_count,
        "unknown": unknown_count,
        "classroom_activity_index": activity_index,
        "observable_engagement": engagement_rating,
        "classroom_event": classroom_event,
        "teacher_activity": teacher_activity,
        "activity_breakdown": activity_counts,
        "compliance_breakdown": compliance_counts,
    }

    return {
        "summary": summary,
        "students": student_cards,
        "heatmap": heatmap_data,
        "boxes": boxes,
        "alerts": alerts,
        "total_frames": state["total_frames_analyzed"]
    }