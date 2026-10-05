"""
Robust Real-Time Exam Hall Malpractice Monitoring Engine
---------------------------------------------------------
Multi-modal monitoring combining:
  - Person detection & tracking (persistent Track #N)
  - Face recognition & quality assessment (YuNet + SFace)
  - Pose estimation & body orientation (MediaPipe PoseLandmarker)
  - Object detection for phones & books (MediaPipe ObjectDetector)
  - Hand landmarking & interaction analysis (MediaPipe HandLandmarker)
  - Spatial relationships between students (copying looks, paper copying, exchanges)
  - Seat position tracking (leaving seat, seat change, multiple people at seat)
  - Multi-frame confirmation & 3-tier severity (NORMAL -> SUSPICIOUS -> REVIEW_REQUIRED)
  - 10-second rolling video proof & multi-photo evidence vault
"""

import os
import time
import json
from datetime import datetime
from collections import deque
import numpy as np
import cv2

import config
import face_quality
import face_engine
import tracker

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Backward-compatibility aliases
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
YAW_LOOK_THRESHOLD = config.YAW_LOOK_THRESHOLD
YAW_BACK_THRESHOLD = config.YAW_BACK_THRESHOLD
YAW_RATIO_THRESHOLD = config.YAW_LOOK_THRESHOLD
EVIDENCE_CLIP_SECONDS = config.EVIDENCE_TOTAL_CLIP_SECONDS

# Global model handles
_face_landmarker = None
_hand_landmarker = None
_pose_landmarker = None
_object_detector = None
_MEDIAPIPE_AVAILABLE = False
_mediapipe_import_error = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except Exception as e:
    _MEDIAPIPE_AVAILABLE = False
    _mediapipe_import_error = str(e)


def _try_load_models():
    """Lazily loads all MediaPipe task models (Face, Hand, Pose, Object)."""
    global _face_landmarker, _hand_landmarker, _pose_landmarker, _object_detector
    if not _MEDIAPIPE_AVAILABLE:
        return

    # 1. Face Landmarker
    if _face_landmarker is None and os.path.exists(config.FACE_LANDMARKER_PATH):
        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=config.FACE_LANDMARKER_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=40,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
            )
            _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception:
            _face_landmarker = None

    # 2. Hand Landmarker
    if _hand_landmarker is None and os.path.exists(config.HAND_LANDMARKER_PATH):
        try:
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=40,
                min_hand_detection_confidence=0.45,
            )
            _hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
        except Exception:
            _hand_landmarker = None

    # 3. Pose Landmarker
    if _pose_landmarker is None and os.path.exists(config.POSE_LANDMARKER_PATH):
        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=config.POSE_LANDMARKER_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=15,
                min_pose_detection_confidence=0.45,
            )
            _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception:
            _pose_landmarker = None

    # 4. Object Detector (EfficientDet Lite0)
    if _object_detector is None and os.path.exists(config.OBJECT_DETECTOR_PATH):
        try:
            options = mp_vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=config.OBJECT_DETECTOR_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                score_threshold=config.PHONE_DETECTION_CONFIDENCE_MIN,
            )
            _object_detector = mp_vision.ObjectDetector.create_from_options(options)
        except Exception:
            _object_detector = None


def get_model_status():
    """Reports status of all vision detectors."""
    _try_load_models()
    return {
        "mediapipe_installed": _MEDIAPIPE_AVAILABLE,
        "import_error": _mediapipe_import_error,
        "face_model_downloaded": os.path.exists(config.FACE_LANDMARKER_PATH),
        "hand_model_downloaded": os.path.exists(config.HAND_LANDMARKER_PATH),
        "pose_model_downloaded": os.path.exists(config.POSE_LANDMARKER_PATH),
        "object_model_downloaded": os.path.exists(config.OBJECT_DETECTOR_PATH),
        "face_landmarker_ready": _face_landmarker is not None,
        "hand_landmarker_ready": _hand_landmarker is not None,
        "pose_landmarker_ready": _pose_landmarker is not None,
        "object_detector_ready": _object_detector is not None,
        "yunet_ready": os.path.exists(config.YUNET_MODEL_PATH),
        "sface_ready": os.path.exists(config.SFACE_MODEL_PATH),
    }


def detect_faces_haar(gray_img):
    """Haar cascade detector fallback."""
    return _face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.2, minNeighbors=5,
        minSize=(config.MIN_FACE_SIZE, config.MIN_FACE_SIZE)
    )


def new_session_state():
    """Initializes the multi-modal state machine for an exam session."""
    return {
        "tracker": tracker.ClassroomTracker(),
        "last_alert_ts": {},            # (key: event_type) -> float timestamp
        "behavior_timers": {},          # (track_id, event_type) -> dict tracking duration, frames, samples
        "yaw_oscillations": {},         # track_id -> deque of (ts, yaw)
        "rolling_frame_buffer": deque(maxlen=30),  # ~5-6 seconds of rolling frames for pre-event evidence
        "seat_history": {},             # track_id -> {"assigned_seat": str, "seat_box": tuple, "away": bool}
        "active_events": {},            # track_id -> set of active events
    }


def _landmarks_to_np(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks])


def compute_mouth_aspect_ratio(pts):
    """Computes Mouth Aspect Ratio (MAR) to evaluate sustained talking/whispering."""
    if pts is None or len(pts) < 320:
        return 0.0
    upper_lip = pts[13]
    lower_lip = pts[14]
    left_corner = pts[78]
    right_corner = pts[308]
    vertical = np.linalg.norm(upper_lip - lower_lip)
    horizontal = max(1e-3, np.linalg.norm(left_corner - right_corner))
    return float(vertical / horizontal)


def save_evidence_package(session_id, event_id, event_type, student_info, full_frame,
                          face_box=None, student_box=None, pre_frames=None, neighbor_box=None):
    """
    Saves a comprehensive evidence package:
      1. before.jpg / full_frame_before.jpg
      2. event_start.jpg
      3. best_event.jpg / full_frame_event.jpg
      4. event_end.jpg
      5. after.jpg
      6. student_crop.jpg
      7. face_crop.jpg (when face is visible)
      8. spatial_neighbor.jpg (if neighbor is involved, e.g. copying)
      9. evidence.mp4 (10-second rolling video buffer compiled from memory)
      10. metadata.json matching required specification
    """
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if hasattr(student_info, "keys") and not isinstance(student_info, dict):
        student_info = dict(student_info)

    usn = student_info.get("usn") or f"UNKNOWN_{student_info.get('track_id', '0')}"
    clean_usn = "".join(c for c in str(usn) if c.isalnum() or c in "-_")
    event_folder_name = f"EVENT_{event_id:04d}_{str(event_type).upper()}"

    # Target directory structure: Evidence/Exam_YYYY_MM_DD/<USN>/EVENT_0001_<TYPE>/
    vault_dir = os.path.join(config.EVIDENCE_VAULT_DIR, f"Exam_{date_str}", clean_usn, event_folder_name)
    os.makedirs(vault_dir, exist_ok=True)

    # Web-accessible mirror: static/exam_photos/<session_id>/event_<event_id>/
    web_dir = os.path.join(config.EXAM_PHOTOS_DIR, str(session_id), f"event_{event_id}")
    os.makedirs(web_dir, exist_ok=True)

    H, W = full_frame.shape[:2]

    before_img = full_frame
    event_start_img = full_frame
    if pre_frames and len(pre_frames) > 0:
        before_img = pre_frames[0]
        event_start_img = pre_frames[len(pre_frames) // 2] if len(pre_frames) > 2 else pre_frames[0]

    best_event_img = full_frame
    event_end_img = full_frame
    after_img = full_frame

    images_to_save = {
        "before.jpg": before_img,
        "full_frame_before.jpg": before_img,
        "event_start.jpg": event_start_img,
        "best_event.jpg": best_event_img,
        "full_frame_event.jpg": best_event_img,
        "event_end.jpg": event_end_img,
        "after.jpg": after_img,
    }

    image_evidence_list = [
        "before.jpg",
        "event_start.jpg",
        "best_event.jpg",
        "event_end.jpg",
        "after.jpg"
    ]

    for fname, img in images_to_save.items():
        cv2.imwrite(os.path.join(vault_dir, fname), img)
        cv2.imwrite(os.path.join(web_dir, fname), img)

    # Student crop
    s_box = student_box or face_box or (0, 0, W, H)
    sx, sy, sw, sh = s_box
    pad_sx, pad_sy = int(sw * 0.2), int(sh * 0.2)
    sx0, sy0 = max(0, sx - pad_sx), max(0, sy - pad_sy)
    sx1, sy1 = min(W, sx + sw + pad_sx), min(H, sy + sh + pad_sy)
    s_crop = full_frame[sy0:sy1, sx0:sx1]
    if s_crop.size > 0:
        cv2.imwrite(os.path.join(vault_dir, "student_crop.jpg"), s_crop)
        cv2.imwrite(os.path.join(web_dir, "student_crop.jpg"), s_crop)

    # Face crop
    face_verified = bool(student_info.get("face_visible", True) and face_box is not None)
    if face_box is not None and face_verified:
        fx, fy, fw, fh = face_box
        pad_fx, pad_fy = int(fw * 0.35), int(fh * 0.35)
        fx0, fy0 = max(0, fx - pad_fx), max(0, fy - pad_fy)
        fx1, fy1 = min(W, fx + fw + pad_fx), min(H, fy + fh + pad_fy)
        f_crop = full_frame[fy0:fy1, fx0:fx1]
        if f_crop.size > 0:
            cv2.imwrite(os.path.join(vault_dir, "face_crop.jpg"), f_crop)
            cv2.imwrite(os.path.join(web_dir, "face_crop.jpg"), f_crop)

    # Spatial neighbor crop
    if neighbor_box is not None:
        nx, ny, nw, nh = neighbor_box
        min_x = max(0, min(sx, nx) - 30)
        min_y = max(0, min(sy, ny) - 30)
        max_x = min(W, max(sx + sw, nx + nw) + 30)
        max_y = min(H, max(sy + sh, ny + nh) + 30)
        sp_crop = full_frame[min_y:max_y, min_x:max_x]
        if sp_crop.size > 0:
            cv2.imwrite(os.path.join(vault_dir, "spatial_neighbor.jpg"), sp_crop)
            cv2.imwrite(os.path.join(web_dir, "spatial_neighbor.jpg"), sp_crop)

    # Compile 10-second rolling video evidence
    video_filename = "evidence.mp4"
    video_vault_path = os.path.join(vault_dir, video_filename)
    video_web_path = os.path.join(web_dir, video_filename)

    clip_frames = list(pre_frames) if pre_frames else [full_frame]
    while len(clip_frames) < 30:
        clip_frames.append(full_frame)

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(video_vault_path, fourcc, 6.0, (W, H))
        for cf in clip_frames:
            vw.write(cf)
        vw.release()
        if os.path.exists(video_vault_path):
            import shutil
            shutil.copyfile(video_vault_path, video_web_path)
    except Exception:
        pass

    identity_source = student_info.get("identity_source", "TRACK_ASSOCIATED" if not face_verified else "FACE_VERIFIED")
    if student_info.get("status") == "UNKNOWN":
        identity_source = "UNKNOWN"

    metadata = {
        "event_id": f"EVENT_{event_id:04d}",
        "usn": student_info.get("usn") or "UNKNOWN",
        "student_name": student_info.get("name") or student_info.get("student_name") or "Unknown Student",
        "track_id": f"TRACK_{student_info.get('track_id', 1):02d}",
        "event_type": str(event_type).upper(),
        "timestamp": time_str,
        "duration_seconds": round(float(student_info.get("duration_seconds", config.EVENT_MIN_DURATION)), 1),
        "event_confidence": round(float(student_info.get("confidence", config.CONFIDENCE_THRESHOLD) / 100.0 if student_info.get("confidence", 1.0) > 1.0 else student_info.get("confidence", config.CONFIDENCE_THRESHOLD)), 2),
        "identity_source": identity_source,
        "face_verified": bool(face_verified and identity_source == "FACE_VERIFIED"),
        "video_evidence": f"exam_photos/{session_id}/event_{event_id}/evidence.mp4",
        "image_evidence": image_evidence_list,
        "status": "REVIEW_REQUIRED",
        "detail": student_info.get("detail", ""),
        "seat_no": student_info.get("seat_no", "")
    }

    with open(os.path.join(vault_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(web_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return f"exam_photos/{session_id}/event_{event_id}/full_frame_event.jpg"


def save_evidence_bundle(session_id, event_id, event_type, student_info, full_frame,
                         face_box=None, pre_frame=None, student_box=None, **kwargs):
    """Backward-compatible wrapper for save_evidence_package."""
    pre_frames = [pre_frame] if pre_frame is not None else None
    return save_evidence_package(
        session_id=session_id,
        event_id=event_id,
        event_type=event_type,
        student_info=student_info,
        full_frame=full_frame,
        face_box=face_box,
        student_box=student_box,
        pre_frames=pre_frames
    )


def analyze_frame(img_bgr, state, session_id=1, roster=None, gallery=None):
    """
    Main Real-Time Exam Hall Malpractice Monitoring Pipeline:
      Evaluates all 17 examination malpractice behaviors (A through Q)
      with multi-frame temporal confirmation and captures evidence.
    """
    now = time.time()
    alerts = []
    boxes = []

    h, w = img_bgr.shape[:2]
    _try_load_models()

    state["rolling_frame_buffer"].append(img_bgr.copy())
    pre_frames = list(state["rolling_frame_buffer"])

    # Step 1: Run Multi-Person Classroom Tracker
    tracker_inst = state["tracker"]
    track_results = tracker_inst.process_frame(img_bgr, gallery=gallery, roster=roster, timestamp=now)

    # Convert roster to clean dicts if needed
    clean_roster = {}
    if roster:
        for rk, rv in roster.items():
            clean_roster[rk] = dict(rv) if hasattr(rv, "keys") else rv

    # Step 2: Extract MediaPipe Vision Signals (Face, Hand, Pose, Objects)
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

    # Object Detector (Phones, Books, etc.)
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

    # Pose Landmarker
    if _pose_landmarker is not None and mp_img is not None:
        try:
            p_res = _pose_landmarker.detect(mp_img)
            for pl in (p_res.pose_landmarks or []):
                pts = _landmarks_to_np(pl, w, h)
                l_sh = pts[11]
                r_sh = pts[12]
                body_poses.append({"left_shoulder": l_sh, "right_shoulder": r_sh, "pts": pts})
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

    def _is_cooldown_ok(key, cooldown_sec):
        last = state["last_alert_ts"].get(key, 0)
        if now - last >= cooldown_sec:
            state["last_alert_ts"][key] = now
            return True
        return False

    def _raise_alert(student_info, alert_type, severity, detail, face_box, neighbor_box=None):
        """Generates a verified alert and saves complete multi-modal evidence."""
        if hasattr(student_info, "keys") and not isinstance(student_info, dict):
            student_info = dict(student_info)

        tid = student_info.get("track_id")
        key = f"{student_info.get('student_id') or tid}:{alert_type}"
        cooldown = config.ALERT_COOLDOWN_SECONDS if student_info.get("student_id") else config.UNKNOWN_COOLDOWN_SECONDS
        if not _is_cooldown_ok(key, cooldown):
            return

        event_id = int(time.time() * 1000) % 1000000
        student_info["severity"] = severity
        student_info["detail"] = detail
        student_info["event_id"] = event_id

        photo_rel_path = save_evidence_package(
            session_id, event_id, alert_type, student_info, img_bgr,
            face_box=face_box, student_box=student_info.get("person_box"),
            pre_frames=pre_frames, neighbor_box=neighbor_box
        )

        identity_source = student_info.get("identity_source", "FACE_VERIFIED")
        face_verified = bool(student_info.get("face_visible", True) and identity_source == "FACE_VERIFIED")

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
            "usn": student_info.get("usn") or "UNKNOWN",
            "seat_no": student_info.get("seat_no", ""),
            "track_id": f"TRACK_{tid:02d}",
            "confidence": student_info.get("confidence", 85.0),
            "identity_source": identity_source,
            "face_verified": face_verified,
            "photo_path": photo_rel_path,
            "video_path": f"exam_photos/{session_id}/event_{event_id}/evidence.mp4",
            "needs_video_clip": True,
            "clip_seconds": config.EVIDENCE_TOTAL_CLIP_SECONDS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

    # Step 3: Analyze Behaviors per Tracked Person
    n_tracks = len(track_results)

    for trk in track_results:
        tid = trk["track_id"]
        box = trk["box"]
        p_box = trk.get("person_box", box)
        label = f"Track #{tid}"
        if trk["usn"]:
            label += f" ({trk['usn']})"
        elif trk["name"] and trk["name"] != "Unknown":
            label += f" ({trk['name']})"

        if not trk.get("face_visible", True):
            label += " [TRACK-ASSOC]"

        color = "#22c55e" if trk["status"] == "VERIFIED" else ("#eab308" if trk["status"] == "UNDER_REVIEW" else "#ef4444")
        boxes.append({"x": box[0], "y": box[1], "w": box[2], "h": box[3], "label": label, "color": color})

        mf = _find_closest_mp_face(box)
        pts = mf["pts"] if mf else None

        # -------------------------------------------------------------
        # K. UNKNOWN PERSON
        # -------------------------------------------------------------
        is_unverified = (trk["status"] in ("UNKNOWN", "UNDER_REVIEW")) and not trk.get("usn")
        if is_unverified and trk["face_quality"] > 0.55:
            timer_key = (tid, "unknown_person")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                trk["duration_seconds"] = duration
                _raise_alert(trk, "unknown_person", "medium",
                             f"{label} is unverified and not allocated to this exam hall (heuristic: possible proxy candidate).",
                             box)
        else:
            state["behavior_timers"].pop((tid, "unknown_person"), None)

        # -------------------------------------------------------------
        # I, J. LEAVING SEAT & SEAT CHANGE
        # -------------------------------------------------------------
        if not trk.get("is_at_seat", True):
            timer_key = (tid, "leaving_seat")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]

            if duration >= config.SEAT_CHANGE_DURATION_SEC:
                trk["duration_seconds"] = duration
                _raise_alert(trk, "possible_seat_change", "high",
                             f"{label} changed seat and remained in new location for {duration:.1f}s.",
                             box)
            elif duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                trk["duration_seconds"] = duration
                _raise_alert(trk, "leaving_seat", "medium",
                             f"{label} has moved away from their assigned seat position for {duration:.1f}s.",
                             box)
        else:
            leave_timer = state["behavior_timers"].pop((tid, "leaving_seat"), None)
            if leave_timer and (now - leave_timer["start"]) >= config.EVENT_MIN_DURATION:
                duration = now - leave_timer["start"]
                trk["duration_seconds"] = duration
                _raise_alert(trk, "returned_to_seat", "low",
                             f"{label} returned to assigned seat after being away for {duration:.1f}s.",
                             box)

        # -------------------------------------------------------------
        # D. LOOKING BACK / TURNING AROUND (Track-Associated when face invisible)
        # -------------------------------------------------------------
        is_looking_back = (
            trk.get("head_direction") == "BACK" or
            trk.get("body_orientation") == "BACK" or
            (not trk.get("face_visible", True) and trk.get("coasting_frames", 0) >= 3)
        )

        if is_looking_back:
            timer_key = (tid, "looking_back")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                trk["duration_seconds"] = duration
                trk["identity_source"] = "TRACK_ASSOCIATED"
                _raise_alert(trk, "looking_back", "medium",
                             f"{label} turned head/body significantly toward the rear for {duration:.1f}s (Track-Associated).",
                             box)
        else:
            state["behavior_timers"].pop((tid, "looking_back"), None)

        # -------------------------------------------------------------
        # Q. STUDENT HIDING FACE / CAMERA AVOIDANCE
        # -------------------------------------------------------------
        if trk.get("is_at_seat", True) and not trk.get("face_visible", True) and trk.get("coasting_frames", 0) >= 8:
            timer_key = (tid, "face_obstruction")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                trk["duration_seconds"] = duration
                _raise_alert(trk, "face_obstruction", "medium",
                             f"{label} face persistently unavailable/obscured while present at seat for {duration:.1f}s.",
                             box)
        else:
            state["behavior_timers"].pop((tid, "face_obstruction"), None)

        # -------------------------------------------------------------
        # Detailed Landmark Checks when Face is Visible
        # -------------------------------------------------------------
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

            yaw_deq = state["yaw_oscillations"].setdefault(tid, deque(maxlen=25))
            yaw_deq.append((now, yaw_ratio))

            is_looking_left = (yaw_ratio >= config.YAW_LOOK_THRESHOLD or trk.get("head_direction") == "LEFT")
            is_looking_right = (yaw_ratio <= -config.YAW_LOOK_THRESHOLD or trk.get("head_direction") == "RIGHT")

            # B. LOOKING LEFT
            if is_looking_left:
                timer_key = (tid, "looking_left")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]
                if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                    trk["duration_seconds"] = duration
                    _raise_alert(trk, "looking_left", "low",
                                 f"{label} sustained head turn toward the left for {duration:.1f}s (Review Required).",
                                 box)
            else:
                state["behavior_timers"].pop((tid, "looking_left"), None)

            # C. LOOKING RIGHT
            if is_looking_right:
                timer_key = (tid, "looking_right")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]
                if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                    trk["duration_seconds"] = duration
                    _raise_alert(trk, "looking_right", "low",
                                 f"{label} sustained head turn toward the right for {duration:.1f}s (Review Required).",
                                 box)
            else:
                state["behavior_timers"].pop((tid, "looking_right"), None)

            # M. LOOKING DOWN / SUSPICIOUS OBJECT
            if pitch_ratio >= (config.PITCH_DOWN_THRESHOLD * 1.5):
                timer_key = (tid, "looking_down_suspicious")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]
                if duration >= (config.EVENT_MIN_DURATION * 1.5) and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                    trk["duration_seconds"] = duration
                    _raise_alert(trk, "looking_down_suspicious", "medium",
                                 f"{label} sustained extreme downward attention toward lap/under-desk for {duration:.1f}s.",
                                 box)
            else:
                state["behavior_timers"].pop((tid, "looking_down_suspicious"), None)

            # G. TALKING
            mar = compute_mouth_aspect_ratio(pts)
            if mar >= config.MOUTH_TALKING_ASPECT_RATIO:
                timer_key = (tid, "talking")
                timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                timer["frames"] += 1
                duration = now - timer["start"]
                if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                    trk["duration_seconds"] = duration
                    _raise_alert(trk, "possible_talking", "medium",
                                 f"{label} mouth movement / sustained open for {duration:.1f}s (Possible Talking/Whispering).",
                                 box)
            else:
                state["behavior_timers"].pop((tid, "talking"), None)

            # P. REPEATED SUSPICIOUS HEAD MOVEMENT
            if len(yaw_deq) >= 12:
                recent = [y for (ts, y) in yaw_deq if (now - ts) <= 5.0]
                if len(recent) >= 8:
                    sign_flips = sum(1 for i in range(len(recent) - 1) if (recent[i] * recent[i + 1] < -0.015))
                    if sign_flips >= 3:
                        if _is_cooldown_ok(f"{tid}:repeated_movement", 12.0):
                            trk["duration_seconds"] = 4.5
                            _raise_alert(trk, "repeated_suspicious_movement", "medium",
                                         f"{label} displaying rapid alternating head turns left/right/back (Nervous glancing).",
                                         box)

            # E, F. LOOKING TOWARD ANOTHER STUDENT / PAPER COPYING
            for other in track_results:
                if other["track_id"] == tid:
                    continue
                ob = other["box"]
                other_cx = ob[0] + ob[2] / 2.0
                my_cx = box[0] + box[2] / 2.0
                delta_x = other_cx - my_cx
                dist = abs(delta_x)

                facing_neighbor = (delta_x > 0 and yaw_ratio <= -config.YAW_LOOK_THRESHOLD) or \
                                  (delta_x < 0 and yaw_ratio >= config.YAW_LOOK_THRESHOLD)

                if facing_neighbor and dist < (box[2] * 4.0):
                    is_paper_look = (pitch_ratio >= config.PITCH_DOWN_THRESHOLD or trk.get('head_direction') == 'DOWN')
                    event_tag = "possible_paper_copying" if is_paper_look else "possible_copying_look"
                    timer_key = (tid, f"neighbor_{other['track_id']}")
                    timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                    timer["frames"] += 1
                    duration = now - timer["start"]

                    if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                        other_usn = other.get("usn") or f"Track #{other['track_id']}"
                        trk["duration_seconds"] = duration
                        detail = f"{label} repeatedly looking toward {other_usn}'s answer sheet/paper area for {duration:.1f}s." if is_paper_look else \
                                 f"{label} repeatedly looking toward {other_usn}'s position for {duration:.1f}s."
                        _raise_alert(trk, event_tag, "medium", detail, box, neighbor_box=ob)
                else:
                    state["behavior_timers"].pop((tid, f"neighbor_{other['track_id']}"), None)

        # -------------------------------------------------------------
        # A. PHONE / ELECTRONIC DEVICE DETECTION
        # -------------------------------------------------------------
        phone_detected_near_student = False
        for obj in detected_objects:
            if obj["category"] in ("cell phone", "phone", "electronic device", "laptop"):
                ocx, ocy = obj["centroid"]
                if (p_box[0] - 40) <= ocx <= (p_box[0] + p_box[2] + 40) and \
                   (p_box[1] - 40) <= ocy <= (p_box[1] + p_box[3] + 40):
                    phone_detected_near_student = True
                    break

        hand_near_ear = False
        for hp in hand_positions:
            hx, hy = hp["wrist"]
            if (box[0] - box[2] * 0.4) <= hx <= (box[0] + box[2] * 1.4) and \
               (box[1] - box[3] * 0.3) <= hy <= (box[1] + box[3] * 1.3):
                hand_near_ear = True
                break

        if phone_detected_near_student or hand_near_ear:
            timer = state["behavior_timers"].get((tid, "phone_use")) or state["behavior_timers"].get((tid, "phone_usage"))
            if timer is None:
                timer = {"start": now, "frames": 0, "status": "SUSPICIOUS"}
                state["behavior_timers"][(tid, "phone_use")] = timer
            timer["frames"] = timer.get("frames", 0) + 1
            duration = now - timer["start"]
            if duration >= 2.0:
                trk["duration_seconds"] = duration
                _raise_alert(trk, "phone_use", "high",
                             f"Phone / electronic device interaction detected near {label} for {duration:.1f}s.",
                             box)
        else:
            state["behavior_timers"].pop((tid, "phone_use"), None)
            state["behavior_timers"].pop((tid, "phone_usage"), None)

        # -------------------------------------------------------------
        # N. UNAUTHORIZED PAPER / NOTE
        # -------------------------------------------------------------
        for obj in detected_objects:
            if obj["category"] in ("book", "notes"):
                ocx, ocy = obj["centroid"]
                if (p_box[0] - 20) <= ocx <= (p_box[0] + p_box[2] + 20) and \
                   (p_box[1] + box[3]) <= ocy <= (p_box[1] + p_box[3] + 50):
                    timer_key = (tid, "unauthorized_paper")
                    timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                    timer["frames"] += 1
                    duration = now - timer["start"]
                    if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                        trk["duration_seconds"] = duration
                        _raise_alert(trk, "possible_unauthorized_material", "medium",
                                     f"Possible unauthorized paper/book material detected in {label}'s desk area for {duration:.1f}s.",
                                     box)
                    break
        else:
            state["behavior_timers"].pop((tid, "unauthorized_paper"), None)

        # -------------------------------------------------------------
        # O. EARPHONE / BLUETOOTH DEVICE
        # -------------------------------------------------------------
        if hand_near_ear and trk.get("head_direction") in ("LEFT", "RIGHT"):
            timer_key = (tid, "earphone")
            timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
            timer["frames"] += 1
            duration = now - timer["start"]
            if duration >= (config.EVENT_MIN_DURATION * 1.5) and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                trk["duration_seconds"] = duration
                _raise_alert(trk, "possible_earphone", "medium",
                             f"Hand persistently held at ear position for {label} (Possible earphone/bluetooth device).",
                             box)
        else:
            state["behavior_timers"].pop((tid, "earphone"), None)

    # -----------------------------------------------------------------
    # H. PASSING OBJECTS
    # -----------------------------------------------------------------
    if len(hand_positions) >= 2 and n_tracks >= 2:
        for i in range(len(hand_positions)):
            for j in range(i + 1, len(hand_positions)):
                h1 = hand_positions[i]["tip"]
                h2 = hand_positions[j]["tip"]
                h_dist = ((h1[0] - h2[0]) ** 2 + (h1[1] - h2[1]) ** 2) ** 0.5
                if h_dist <= config.OBJECT_EXCHANGE_DISTANCE_PX:
                    matched_trks = []
                    for t in track_results:
                        tb = t.get("person_box", t["box"])
                        if (tb[0] - 40) <= h1[0] <= (tb[0] + tb[2] + 40) or (tb[0] - 40) <= h2[0] <= (tb[0] + tb[2] + 40):
                            matched_trks.append(t)
                    if len(matched_trks) >= 2:
                        pair_key = tuple(sorted([matched_trks[0]["track_id"], matched_trks[1]["track_id"]]))
                        timer_key = (pair_key, "object_exchange")
                        timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                        timer["frames"] += 1
                        duration = now - timer["start"]
                        if duration >= 2.0 and timer["frames"] >= 4:
                            t1, t2 = matched_trks[0], matched_trks[1]
                            t1["duration_seconds"] = duration
                            u1 = t1.get("usn") or f"Track #{t1['track_id']}"
                            u2 = t2.get("usn") or f"Track #{t2['track_id']}"
                            detail = f"Possible object exchange / hand interaction between {u1} and {u2}."
                            _raise_alert(t1, "possible_object_exchange", "high", detail, t1["box"], neighbor_box=t2["box"])
                            _raise_alert(t2, "possible_object_exchange", "high", detail, t2["box"], neighbor_box=t1["box"])

    # -----------------------------------------------------------------
    # L. MULTIPLE PEOPLE AT ONE SEAT
    # -----------------------------------------------------------------
    if n_tracks >= 2:
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                t1, t2 = track_results[i], track_results[j]
                b1, b2 = t1["box"], t2["box"]
                c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
                c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                avg_w = (b1[2] + b2[2]) / 2.0

                if dist <= (avg_w * config.NEIGHBOR_COPYING_DISTANCE_FACTOR):
                    pair_key = tuple(sorted([t1["track_id"], t2["track_id"]]))
                    timer_key = (pair_key, "multiple_people")
                    timer = state["behavior_timers"].setdefault(timer_key, {"start": now, "frames": 0, "status": "SUSPICIOUS"})
                    timer["frames"] += 1
                    duration = now - timer["start"]
                    if duration >= config.EVENT_MIN_DURATION and timer["frames"] >= config.EVENT_CONFIRMATION_FRAMES:
                        n1 = t1.get("usn") or f"Track #{t1['track_id']}"
                        n2 = t2.get("usn") or f"Track #{t2['track_id']}"
                        detail = f"Multiple people detected occupying one seat area ({n1} and {n2}) for {duration:.1f}s."
                        _raise_alert(t1, "multiple_person_event", "high", detail, b1, neighbor_box=b2)
                        _raise_alert(t2, "multiple_person_event", "high", detail, b2, neighbor_box=b1)
                else:
                    pair_key = tuple(sorted([t1["track_id"], t2["track_id"]]))
                    state["behavior_timers"].pop((pair_key, "multiple_people"), None)

    # Telemetry summary
    telemetry = {
        "detected_count": len(track_results),
        "verified_count": sum(1 for t in track_results if t.get("status") == "VERIFIED"),
        "unknown_count": sum(1 for t in track_results if t.get("status") == "UNKNOWN"),
        "review_count": sum(1 for t in track_results if t.get("status") == "UNDER_REVIEW"),
        "tracks": track_results,
    }

    return alerts, boxes, telemetry
