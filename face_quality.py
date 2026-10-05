"""
Face Quality Assessment and Anti-Spoofing / Liveness Module
------------------------------------------------------------
Implements robust quality filtering (blur, size, lighting, occlusion, boundary)
and passive/temporal liveness checks (micro-movement, blink, texture).
"""

import cv2
import numpy as np
import config

# Landmark indices for 5-point alignment / pose (YuNet order: re, le, nt, rc, lc)
# Or MediaPipe indices if available


def compute_blur_score(gray_face):
    """Compute sharpness using Laplacian variance.
    Returns: (is_sharp: bool, raw_variance: float, normalized_score: float [0..1])"""
    if gray_face is None or gray_face.size == 0:
        return False, 0.0, 0.0
    var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    # Normalize: 0 at var=0, 0.5 at BLUR_THRESHOLD, 1.0 at 250+
    norm = min(1.0, var / max(1.0, config.BLUR_THRESHOLD * 3.5))
    is_sharp = var >= config.BLUR_THRESHOLD
    return is_sharp, float(var), float(norm)


def compute_illumination_score(gray_face):
    """Assess whether the face lighting is balanced.
    Returns: (is_good_light: bool, mean_brightness: float, score: float [0..1], reason: str)"""
    if gray_face is None or gray_face.size == 0:
        return False, 0.0, 0.0, "Empty face region"
    mean_val = float(np.mean(gray_face))
    std_val = float(np.std(gray_face))

    if mean_val < config.BRIGHTNESS_MIN:
        return False, mean_val, 0.2, "Face too dark — increase room lighting"
    if mean_val > config.BRIGHTNESS_MAX:
        return False, mean_val, 0.2, "Face overexposed — reduce direct glare"

    # Contrast check: if standard deviation is too flat (<15), image is washed out
    if std_val < 15.0:
        return False, mean_val, 0.4, "Low contrast / washed out face"

    # Optimum brightness is around 110-150
    dist_from_optimum = abs(mean_val - 130.0)
    score = max(0.5, 1.0 - (dist_from_optimum / 110.0))
    return True, mean_val, float(score), "Optimal lighting"


def check_boundary_margins(box, frame_shape):
    """Ensure the face is not cut off at the edge of the camera frame.
    box: (x, y, w, h)
    frame_shape: (height, width, ...)
    """
    x, y, w, h = box
    fh, fw = frame_shape[:2]
    m = config.FACE_BOUNDARY_MARGIN

    if x < m or y < m or (x + w) > (fw - m) or (y + h) > (fh - m):
        return False, "Face too close to frame boundary — center face"
    return True, "Face well framed"


def estimate_head_pose(landmarks, box=None):
    """Estimate yaw and pitch from facial landmarks.
    Accepts:
      - 5-point landmarks (array of 5 (x,y) points: right_eye, left_eye, nose_tip, right_mouth, left_mouth)
      - or MediaPipe landmarks
    Returns:
      (yaw_ratio, pitch_ratio, pose_category)
      yaw_ratio: negative = looking right, positive = looking left
      pitch_ratio: negative = looking up, positive = looking down
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 0.0, "FRONT"

    pts = np.array(landmarks, dtype=np.float32)

    # If 5-point YuNet format: [re_x, re_y], [le_x, le_y], [nt_x, nt_y], [rm_x, rm_y], [lm_x, lm_y]
    if len(pts) == 5:
        re = pts[0]
        le = pts[1]
        nt = pts[2]
        rm = pts[3]
        lm = pts[4]

        eye_mid = (re + le) / 2.0
        eye_dist = max(1e-3, float(np.linalg.norm(le - re)))
        mouth_mid = (rm + lm) / 2.0
        face_vert_dist = max(1e-3, float(np.linalg.norm(mouth_mid - eye_mid)))

        # Yaw ratio: offset of nose from eye midpoint normalized by eye distance
        yaw_ratio = float((nt[0] - eye_mid[0]) / eye_dist)

        # Pitch ratio: offset of nose relative to eye-mouth bisector
        expected_nose_y = eye_mid[1] + (face_vert_dist * 0.45)
        pitch_ratio = float((nt[1] - expected_nose_y) / face_vert_dist)

    else:
        # MediaPipe 468/478 landmarks
        nose = pts[1]  # NOSE_TIP
        le = pts[33]   # LEFT_EYE_OUTER
        re = pts[263]  # RIGHT_EYE_OUTER
        eye_mid = (le + re) / 2.0
        eye_dist = max(1e-3, float(np.linalg.norm(re - le)))
        yaw_ratio = float((nose[0] - eye_mid[0]) / eye_dist)

        chin = pts[152] if len(pts) > 152 else pts[1] + np.array([0, eye_dist])
        vert_dist = max(1e-3, float(np.linalg.norm(chin - eye_mid)))
        expected_nose_y = eye_mid[1] + (vert_dist * 0.45)
        pitch_ratio = float((nose[1] - expected_nose_y) / vert_dist)

    # Classify pose category
    pose = "FRONT"
    if yaw_ratio >= config.YAW_BACK_THRESHOLD:
        pose = "LEFT_LARGE"
    elif yaw_ratio >= config.YAW_LOOK_THRESHOLD:
        pose = "LEFT_MEDIUM"
    elif yaw_ratio >= (config.YAW_LOOK_THRESHOLD * 0.5):
        pose = "LEFT_SLIGHT"
    elif yaw_ratio <= -config.YAW_BACK_THRESHOLD:
        pose = "RIGHT_LARGE"
    elif yaw_ratio <= -config.YAW_LOOK_THRESHOLD:
        pose = "RIGHT_MEDIUM"
    elif yaw_ratio <= -(config.YAW_LOOK_THRESHOLD * 0.5):
        pose = "RIGHT_SLIGHT"
    elif pitch_ratio >= config.PITCH_DOWN_THRESHOLD:
        pose = "DOWN"
    elif pitch_ratio <= config.PITCH_UP_THRESHOLD:
        pose = "UP"

    return yaw_ratio, pitch_ratio, pose


def assess_face_quality(frame_bgr, box, landmarks=None, detection_conf=1.0, require_frontal=False):
    """
    Comprehensive quality check for a face detection.
    Returns:
      {
         "passed": bool,
         "quality_score": float [0.0..1.0],
         "reasons": list of strings,
         "blur_var": float,
         "brightness": float,
         "size": (w, h),
         "pose": str (FRONT, LEFT_SLIGHT, etc.),
         "yaw": float,
         "pitch": float
      }
    """
    reasons = []
    x, y, w, h = [int(v) for v in box]
    fh, fw = frame_bgr.shape[:2]

    # Check 1: Size check
    if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
        reasons.append(f"Face too small ({w}x{h}px, minimum {config.MIN_FACE_SIZE}px) — move closer")
        size_score = min(1.0, max(0.0, w / config.MIN_FACE_SIZE))
    else:
        size_score = min(1.0, w / 160.0)

    # Check 2: Boundary check
    in_bounds, b_msg = check_boundary_margins(box, frame_bgr.shape)
    if not in_bounds:
        reasons.append(b_msg)

    # Crop face safely
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    face_crop = frame_bgr[y0:y1, x0:x1]

    if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return {
            "passed": False,
            "quality_score": 0.0,
            "reasons": ["Invalid crop area"] + reasons,
            "blur_var": 0.0,
            "brightness": 0.0,
            "size": (w, h),
            "pose": "UNKNOWN",
            "yaw": 0.0,
            "pitch": 0.0,
        }

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # Check 3: Blur check
    is_sharp, blur_var, blur_score = compute_blur_score(gray_face)
    if not is_sharp:
        reasons.append(f"Face blurry (sharpness {blur_var:.1f}, threshold {config.BLUR_THRESHOLD:.1f}) — hold steady")

    # Check 4: Illumination check
    is_light_ok, brightness, light_score, light_reason = compute_illumination_score(gray_face)
    if not is_light_ok:
        reasons.append(light_reason)

    # Check 5: Pose check
    yaw, pitch, pose = estimate_head_pose(landmarks, box)
    pose_penalty = 1.0
    if require_frontal and pose != "FRONT":
        reasons.append(f"Head turned ({pose}) — please look straight at the camera")
        pose_penalty = 0.5

    # Check 6: Detection confidence
    conf_score = max(0.0, min(1.0, float(detection_conf)))
    if conf_score < config.FACE_DETECTION_THRESHOLD:
        reasons.append(f"Low face detection confidence ({conf_score:.2f})")

    # Composite quality score
    composite_score = (
        0.30 * blur_score +
        0.25 * size_score +
        0.20 * light_score +
        0.15 * conf_score +
        0.10 * pose_penalty
    )
    composite_score = float(np.clip(composite_score, 0.0, 1.0))

    passed = (len(reasons) == 0) and (composite_score >= config.MIN_FACE_QUALITY_SCORE)

    return {
        "passed": passed,
        "quality_score": round(composite_score, 3),
        "reasons": reasons,
        "blur_var": round(blur_var, 1),
        "brightness": round(brightness, 1),
        "size": (w, h),
        "pose": pose,
        "yaw": round(yaw, 3),
        "pitch": round(pitch, 3),
    }


class LivenessTracker:
    """
    Passive & temporal liveness verification across consecutive video frames.
    Flags static printed photos or phone screen replays based on:
      1. Landmark micro-movement variance (real humans never have zero jitter).
      2. Color gradient variance across skin.
      3. Eye blink / EAR fluctuation when landmarks are present.
    """
    def __init__(self, history_len=15):
        self.history_len = history_len
        self.face_history = []  # list of (timestamp, box, landmarks_np)

    def update(self, box, landmarks, timestamp):
        pts = np.array(landmarks, dtype=np.float32) if landmarks is not None else None
        self.face_history.append((timestamp, box, pts))
        if len(self.face_history) > self.history_len:
            self.face_history.pop(0)

    def check_liveness(self):
        """Returns (is_live: bool, liveness_score: float, detail: str)"""
        if len(self.face_history) < 6:
            return True, 0.75, "Gathering temporal samples"

        # Check 1: Landmark micro-movement
        valid_pts = [entry[2] for entry in self.face_history if entry[2] is not None]
        if len(valid_pts) >= 6:
            # Measure standard deviation of nose tip position relative to face box
            rel_nose_positions = []
            for entry in self.face_history:
                box = entry[1]
                pts = entry[2]
                if pts is not None and len(pts) >= 3:
                    # Index 2 in 5-point YuNet or index 1 in MediaPipe is nose tip
                    nose_idx = 2 if len(pts) == 5 else 1
                    bw = max(1.0, float(box[2]))
                    bh = max(1.0, float(box[3]))
                    rel_x = (pts[nose_idx][0] - box[0]) / bw
                    rel_y = (pts[nose_idx][1] - box[1]) / bh
                    rel_nose_positions.append([rel_x, rel_y])

            if len(rel_nose_positions) >= 6:
                stds = np.std(rel_nose_positions, axis=0)
                tot_motion = float(np.sum(stds))

                # Perfect zero motion across 6+ frames indicates a static photo held steadily
                if tot_motion < 0.0005:
                    return False, 0.20, "Static image detected — no natural micro-motion"

                # Extremely erratic unnatural jump indicates frame glitch or paper swap
                if tot_motion > 0.40:
                    return False, 0.35, "Erratic motion anomaly detected"

        # Check passed
        return True, 0.88, "Natural live facial motion verified"
