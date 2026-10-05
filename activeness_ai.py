"""
Class Activeness Monitoring (replaces Active Participation Tracking, 9.3)
--------------------------------------------------------------------------
Heuristic, webcam-only "how engaged was the class" signals, tuned for
classroom-camera conditions (20-60 faces, small, distant, moving) --
same camera assumptions as the participation module this replaces.
Reuses the landmarker models already downloaded by download_models.py;
no new model files are required.

WHAT THIS MEASURES, AND HOW HONESTLY IT MEASURES IT
--------------------------------------------------------------------------
Per sampled frame, once a student's face has been tracked long enough
to say anything at all, they're placed in exactly one band:

  active        -> face mostly pointed at the front of the room
  turned_away   -> head mostly turned away from the front (heuristic:
                   possibly distracted / talking to a neighbour --
                   NOT proof of anything)
  eyes_closed   -> eyes measured shut for several CONSECUTIVE sampled
                   frames in a row (heuristic: possible drowsiness --
                   a single closed-eye frame is almost always a blink,
                   not sleep, so this requires sustained closure)
  sleeping      -> eyes_closed held for a long WALL-CLOCK stretch
                   (default 5 minutes, see SLEEP_DURATION_SECONDS) --
                   this is the "head down / asleep" tier a short blink
                   or a slow eye-rub can never reach. When the head-pose
                   reading also leans "bowed downward" for the same
                   stretch that goes in the alert text as corroboration,
                   but eyes-closed duration is always the trigger -- 2D
                   pitch-from-landmarks alone is too unreliable at
                   classroom-camera distance to fire an alert by itself
  phone_use     -> a hand lingering near the ear/face for several
                   consecutive sampled frames (heuristic: possible
                   phone use -- this is NOT object detection of an
                   actual phone, exactly like exam_ai.py's signal of
                   the same name)
  phone_use_sustained -> phone_use held for a long WALL-CLOCK stretch
                   (default 20s, see PHONE_SUSTAINED_SECONDS) -- a
                   quick check of the time is common and mostly
                   harmless; a hand staying near the face for a
                   sustained stretch is a stronger "worth a look" signal
                   and is treated as a higher-severity warning

"sleeping" and "phone_use_sustained" are the two ESCALATED bands: each
gets its own evidence photo (independent alert-cooldown key from the
plain eyes_closed/phone_use signal) and the app layer asks the browser
to also upload a short (5-10s) buffered video clip as extra context --
see EVIDENCE_CLIP_SECONDS and the `needs_video_clip` flag on alerts.

A running per-student tally of how many scored samples fell in each
band is kept for the whole session, which is what lets the session
report say "engaged in ~86% of checks this period" -- that phrasing
matters: it is the fraction of heuristic samples that looked engaged,
not a certified measurement of attention, and every place this number
is displayed must say so.

ACCURACY MEASURES (same philosophy as the module this replaces)
--------------------------------------------------------------------------
1. STABLE PER-STUDENT IDENTITY across frames via IoU face tracking --
   without it, "student #3" silently becomes a different physical
   student every frame and all counts are meaningless.
2. DEBOUNCED eyes-closed and phone-use (require several consecutive
   sampled frames, not one) -- a single frame of "eyes shut" is a
   blink far more often than it's drowsiness, and a single frame of
   "hand near face" is scratching an ear far more often than a phone.
   Requiring persistence filters almost all of that out.
3. ROLLING-WINDOW MAJORITY VOTE for head orientation, not a
   single-frame verdict -- landmark jitter alone can flip a frame's
   yaw reading.
4. MINIMUM FACE-SIZE GATE -- faces too small/distant to score
   reliably are excluded rather than guessed at.
5. NOTHING IS SCORED until a track has enough samples to say anything
   at all ("insufficient_data" is not the same as "inactive").
6. NO SINGLE NUMBER CLAIMS TO BE "HOW ATTENTIVE THIS STUDENT WAS" --
   every output is a sample-based percentage across explicit,
   named heuristic bands, always shown with what it is and isn't.

This is a teaching aid meant to help a teacher notice classroom-wide
and per-student PATTERNS worth a look -- not an automatic verdict on
any individual student, not a disciplinary record, and not something
that should be handed to anyone as proof of what a specific student
was doing at a specific moment. Vision-only detection at classroom
distance has a real false-positive rate (head down is not the same as
asleep; a hand near an ear is not the same as a phone) -- treat every
alert and every percentage here as "worth a look," never as fact.
"""

import time
from collections import deque

import numpy as np
import cv2

import exam_ai as _mp_ai  # reuse loaded landmarkers + cascade

# ---------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------
MIN_FACE_SIZE_PX = 55            # faces smaller than this (px width) are skipped
IOU_MATCH_THRESHOLD = 0.3        # min overlap to treat two boxes as "same track"
TRACK_LOST_SECONDS = 4           # drop a track if not matched for this long
ORIENTATION_WINDOW = 12          # rolling samples kept for head-orientation trend
ORIENTATION_AWAY_FRACTION = 0.6  # fraction of window that must agree to call a trend
YAW_RATIO_THRESHOLD = _mp_ai.YAW_RATIO_THRESHOLD  # reuse same yaw definition as exam_ai
IDENTITY_RECHECK_SECONDS = 20    # how often to re-run the (expensive) recognizer per track

EAR_CLOSED_THRESHOLD = 0.19          # eye-aspect-ratio below this = "eyes shut" this frame
EYES_CLOSED_DEBOUNCE_SAMPLES = 3     # consecutive sampled frames eyes must stay shut
PHONE_NEAR_FACE_MARGIN = 0.6         # how far around a face box counts as "near", as in exam_ai
PHONE_DEBOUNCE_SAMPLES = 2           # consecutive sampled frames hand must stay near face

ALERT_COOLDOWN_SECONDS = 45      # don't save another evidence photo for the same
                                  # student+behaviour too often -- these are heuristic
                                  # signals, not confirmed events, so a flood of
                                  # near-identical photos wouldn't add real evidence

# --- escalation thresholds: short heuristic blip vs. sustained pattern ---
SLEEP_DURATION_SECONDS = 300     # eyes-closed held continuously this long -> "sleeping"
                                  # (5 min default; tune here, not a hard classroom rule)
PHONE_SUSTAINED_SECONDS = 20     # phone_use held continuously this long -> "phone_use_sustained"
ESCALATED_ALERT_COOLDOWN_SECONDS = 120  # separate (longer) cooldown for the escalated
                                  # bands, so a still-ongoing sleeping/phone stretch
                                  # doesn't spam a fresh photo+video every 45s
EVIDENCE_CLIP_SECONDS = 8        # requested length of the short video clip the browser
                                  # buffers and uploads for an escalated alert (5-10s)

# Face-mesh landmark indices (468-point topology) for eye-aspect-ratio
LEFT_EYE_H = (33, 133)
LEFT_EYE_V = ((159, 145), (158, 153))
RIGHT_EYE_H = (362, 263)
RIGHT_EYE_V = ((386, 374), (385, 380))

# Rough 2D head-pitch proxy: how far down the nose sits between the
# forehead line and the chin. NOT a real 3D head-pose estimate (that
# would need the model's depth output and per-camera calibration) --
# just a corroborating hint used in the "sleeping" alert text, never
# a trigger by itself (see module docstring).
FOREHEAD_TOP = 10
CHIN_BOTTOM = 152
PITCH_DOWN_RATIO = 0.62          # nose sitting past this fraction toward the chin
                                  # reads as "head bowed downward" for that frame


def _pitch_down(pts):
    """True if the nose landmark sits unusually close to the chin relative
    to the forehead-to-chin span this frame -- a rough "looking down"
    hint. Returns None if the span is degenerate (face edge-on, etc.)."""
    fy = pts[FOREHEAD_TOP][1]
    cy = pts[CHIN_BOTTOM][1]
    ny = pts[_mp_ai.NOSE_TIP][1]
    span = cy - fy
    if span <= 1e-3:
        return None
    return ((ny - fy) / span) >= PITCH_DOWN_RATIO


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _ear(pts, h_idx, v_idx_pairs):
    """Eye-aspect-ratio for one eye: mean vertical eyelid gap / horizontal
    eye width. Lower = more closed. Standard formulation, just with a
    variable number of vertical pairs so both eyes can share the code."""
    lx, ly = pts[h_idx[0]]
    rx, ry = pts[h_idx[1]]
    horiz = np.hypot(rx - lx, ry - ly)
    if horiz <= 1e-3:
        return None
    vert_total = 0.0
    for (top_i, bot_i) in v_idx_pairs:
        tx, ty = pts[top_i]
        bx, by = pts[bot_i]
        vert_total += np.hypot(bx - tx, by - ty)
    return (vert_total / len(v_idx_pairs)) / horiz


def new_session_state():
    """Fresh per-session tracking state (mirrors exam_ai's pattern)."""
    return {
        "tracks": {},        # track_id -> track dict (see _new_track)
        "next_track_id": 1,
    }


def _new_track(box, now):
    return {
        "box": box,
        "last_seen_ts": now,
        "student_id": None,
        "student_name": None,
        "last_identify_ts": 0.0,
        "orientation_history": deque(maxlen=ORIENTATION_WINDOW),
        "eyes_closed_streak": 0,
        "eyes_closed_confirmed": False,
        "eyes_closed_since_ts": None,   # wall-clock start of the current confirmed streak
        "pitch_history": deque(maxlen=8),
        "phone_near_streak": 0,
        "phone_confirmed": False,
        "phone_confirmed_since_ts": None,
        "band_counts": {"active": 0, "turned_away": 0, "eyes_closed": 0,
                         "sleeping": 0, "phone_use": 0, "phone_use_sustained": 0},
        "total_scored": 0,
        "last_alert_ts": {},
    }


def _match_tracks(state, faces, now):
    """Match this frame's detected face boxes to existing tracks via IoU."""
    unmatched_box_idxs = list(range(len(faces)))
    matched_pairs = []

    for track_id, track in state["tracks"].items():
        best_idx, best_iou = None, 0.0
        for idx in unmatched_box_idxs:
            score = _iou(track["box"], faces[idx])
            if score > best_iou:
                best_idx, best_iou = idx, score
        if best_idx is not None and best_iou >= IOU_MATCH_THRESHOLD:
            matched_pairs.append((track_id, faces[best_idx]))
            unmatched_box_idxs.remove(best_idx)

    for idx in unmatched_box_idxs:
        track_id = state["next_track_id"]
        state["next_track_id"] += 1
        state["tracks"][track_id] = _new_track(faces[idx], now)
        matched_pairs.append((track_id, faces[idx]))

    stale = []
    seen_ids = {tid for tid, _ in matched_pairs}
    for track_id, track in state["tracks"].items():
        if track_id in seen_ids:
            box = next(b for tid, b in matched_pairs if tid == track_id)
            track["box"] = box
            track["last_seen_ts"] = now
        elif now - track["last_seen_ts"] > TRACK_LOST_SECONDS:
            stale.append(track_id)
    for track_id in stale:
        del state["tracks"][track_id]

    return matched_pairs


def _cooldown_ok(track, key, now, cooldown=ALERT_COOLDOWN_SECONDS):
    last = track["last_alert_ts"].get(key, 0)
    if now - last >= cooldown:
        track["last_alert_ts"][key] = now
        return True
    return False


def analyze_frame(img_bgr, state, identify_face_fn=None, save_evidence_fn=None):
    """
    Run every activeness detector against one (sampled, throttled) frame.

    identify_face_fn: optional callable(gray_face_crop) -> (student_id,
    student_name) or (None, None). Wire this to the app's existing LBPH
    recognizer used for attendance.

    save_evidence_fn: optional callable(img_bgr, (x,y,w,h), alert_type,
    student_id, student_name) -> photo_path (str) or None. Called only
    when a disengagement signal is first confirmed for a track (subject
    to ALERT_COOLDOWN_SECONDS), so it doesn't flood storage.

    Returns (results, alerts):
      results = per-track dicts for the live view: {track_id,
        student_id, student_name, box, orientation_trend, band,
        band_counts, total_scored}
      alerts  = list of dicts ready to persist: {track_id, student_id,
        student_name, alert_type, detail, photo_path, created_at_ts}
    """
    now = time.time()
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = [tuple(int(v) for v in f) for f in _mp_ai.detect_faces_haar(gray)]
    scorable_faces = [f for f in faces if f[2] >= MIN_FACE_SIZE_PX]

    pairs = _match_tracks(state, scorable_faces, now)

    _mp_ai._try_load_landmarkers()
    results = []
    alerts = []

    face_landmarks_list = []
    hand_landmarks_list = []
    if _mp_ai._face_landmarker is not None and scorable_faces:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = _mp_ai.mp.Image(image_format=_mp_ai.mp.ImageFormat.SRGB, data=rgb)
            face_result = _mp_ai._face_landmarker.detect(mp_image)
            face_landmarks_list = face_result.face_landmarks or []
        except Exception:
            face_landmarks_list = []
    if _mp_ai._hand_landmarker is not None and scorable_faces:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = _mp_ai.mp.Image(image_format=_mp_ai.mp.ImageFormat.SRGB, data=rgb)
            hand_result = _mp_ai._hand_landmarker.detect(mp_image)
            hand_landmarks_list = hand_result.hand_landmarks or []
        except Exception:
            hand_landmarks_list = []

    wrist_points = []
    for hand in hand_landmarks_list:
        wrist = hand[0]
        wrist_points.append((wrist.x * w, wrist.y * h))

    for track_id, box in pairs:
        track = state["tracks"][track_id]
        x, y, fw, fh = box

        # --- periodic identity resolution (optional, expensive) ---
        if identify_face_fn is not None and (now - track["last_identify_ts"]) >= IDENTITY_RECHECK_SECONDS:
            try:
                crop = gray[max(0, y):y + fh, max(0, x):x + fw]
                if crop.size > 0:
                    sid, sname = identify_face_fn(crop)
                    if sid is not None:
                        track["student_id"] = sid
                        track["student_name"] = sname
                track["last_identify_ts"] = now
            except Exception:
                pass

        # --- find this face's mediapipe landmarks, if any ---
        pts = None
        for face_landmarks in face_landmarks_list:
            cand = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])
            cx, cy = cand[:, 0].mean(), cand[:, 1].mean()
            if x <= cx <= x + fw and y <= cy <= y + fh:
                pts = cand
                break

        # --- head orientation (rolling majority vote, same as before) ---
        facing_forward = None
        if pts is not None:
            nose = pts[_mp_ai.NOSE_TIP]
            left_eye = pts[_mp_ai.LEFT_EYE_OUTER]
            right_eye = pts[_mp_ai.RIGHT_EYE_OUTER]
            eye_mid = (left_eye + right_eye) / 2.0
            eye_dist = np.linalg.norm(left_eye - right_eye)
            if eye_dist > 1e-3:
                yaw_ratio = (nose[0] - eye_mid[0]) / eye_dist
                facing_forward = abs(yaw_ratio) < YAW_RATIO_THRESHOLD

        if facing_forward is not None:
            track["orientation_history"].append(facing_forward)

        orientation_trend = "insufficient_data"
        if len(track["orientation_history"]) >= max(4, ORIENTATION_WINDOW // 2):
            away_fraction = 1 - (sum(track["orientation_history"]) / len(track["orientation_history"]))
            if away_fraction >= ORIENTATION_AWAY_FRACTION:
                orientation_trend = "mostly_turned_away"
            elif away_fraction <= (1 - ORIENTATION_AWAY_FRACTION):
                orientation_trend = "mostly_facing_forward"
            else:
                orientation_trend = "mixed"

        # --- eyes-closed (debounced across consecutive samples) ---
        eyes_closed_now = False
        if pts is not None:
            left_ear = _ear(pts, LEFT_EYE_H, LEFT_EYE_V)
            right_ear = _ear(pts, RIGHT_EYE_H, RIGHT_EYE_V)
            ear_vals = [v for v in (left_ear, right_ear) if v is not None]
            if ear_vals:
                eyes_closed_now = (sum(ear_vals) / len(ear_vals)) < EAR_CLOSED_THRESHOLD

        if eyes_closed_now:
            track["eyes_closed_streak"] += 1
        else:
            track["eyes_closed_streak"] = 0
            track["eyes_closed_confirmed"] = False
            track["eyes_closed_since_ts"] = None
        if track["eyes_closed_streak"] >= EYES_CLOSED_DEBOUNCE_SAMPLES:
            if not track["eyes_closed_confirmed"]:
                track["eyes_closed_since_ts"] = now  # streak just became "confirmed"
            track["eyes_closed_confirmed"] = True

        # rough head-pitch hint for this frame, used only as corroborating
        # text on a "sleeping" alert -- never a trigger by itself
        if pts is not None:
            pd = _pitch_down(pts)
            if pd is not None:
                track["pitch_history"].append(pd)
        pitch_mostly_down = (
            len(track["pitch_history"]) >= 4
            and (sum(track["pitch_history"]) / len(track["pitch_history"])) >= 0.6
        )

        eyes_closed_duration = (
            (now - track["eyes_closed_since_ts"]) if track["eyes_closed_since_ts"] else 0
        )
        is_sleeping = track["eyes_closed_confirmed"] and eyes_closed_duration >= SLEEP_DURATION_SECONDS

        # --- phone-use: a hand lingering near this face (debounced) ---
        phone_near_now = False
        mx, my = fw * PHONE_NEAR_FACE_MARGIN, fh * PHONE_NEAR_FACE_MARGIN
        for wx, wy in wrist_points:
            if (x - mx) <= wx <= (x + fw + mx) and (y - my) <= wy <= (y + fh + my):
                phone_near_now = True
                break

        if phone_near_now:
            track["phone_near_streak"] += 1
        else:
            track["phone_near_streak"] = 0
            track["phone_confirmed"] = False
            track["phone_confirmed_since_ts"] = None
        if track["phone_near_streak"] >= PHONE_DEBOUNCE_SAMPLES:
            if not track["phone_confirmed"]:
                track["phone_confirmed_since_ts"] = now
            track["phone_confirmed"] = True

        phone_duration = (
            (now - track["phone_confirmed_since_ts"]) if track["phone_confirmed_since_ts"] else 0
        )
        is_phone_sustained = track["phone_confirmed"] and phone_duration >= PHONE_SUSTAINED_SECONDS

        # --- band decision + evidence photo/video on confirmation ---
        # Escalated bands (sleeping / phone_use_sustained) take priority
        # over their short-form counterparts -- once a stretch has run
        # long enough to escalate, that's the more useful thing to show.
        band = "insufficient_data"
        if orientation_trend != "insufficient_data":
            if is_phone_sustained:
                band = "phone_use_sustained"
            elif track["phone_confirmed"]:
                band = "phone_use"
            elif is_sleeping:
                band = "sleeping"
            elif track["eyes_closed_confirmed"]:
                band = "eyes_closed"
            elif orientation_trend == "mostly_turned_away":
                band = "turned_away"
            else:
                band = "active"  # mostly_facing_forward or mixed -- benefit of the doubt

            # sleeping/phone_use_sustained roll up into the same tally
            # bucket as their short-form counterpart for the session's
            # "active so far" percentage math, plus their own count
            tally_band = band
            track["band_counts"][tally_band] = track["band_counts"].get(tally_band, 0) + 1
            track["total_scored"] += 1

            escalated = band in ("sleeping", "phone_use_sustained")
            cooldown_key = band
            cooldown = ESCALATED_ALERT_COOLDOWN_SECONDS if escalated else ALERT_COOLDOWN_SECONDS

            if band in ("phone_use", "eyes_closed", "turned_away", "sleeping", "phone_use_sustained") \
                    and _cooldown_ok(track, cooldown_key, now, cooldown):
                photo_path = None
                if save_evidence_fn is not None:
                    try:
                        photo_path = save_evidence_fn(
                            img_bgr, box, band, track["student_id"], track["student_name"]
                        )
                    except Exception:
                        photo_path = None
                detail = {
                    "phone_use": "Hand lingering near ear/face for several consecutive checks "
                                  "(heuristic: possible phone use -- not confirmed).",
                    "phone_use_sustained": f"Hand has stayed near ear/face continuously for "
                                  f"~{int(phone_duration)}s (heuristic: sustained possible "
                                  f"phone use -- not confirmed; not object-detection of a phone).",
                    "eyes_closed": "Eyes measured shut for several consecutive checks "
                                   "(heuristic: possible drowsiness -- not confirmed).",
                    "sleeping": (
                        f"Eyes measured shut continuously for ~{int(eyes_closed_duration // 60)} "
                        f"min {int(eyes_closed_duration % 60)}s (heuristic: possible sleeping -- "
                        f"not confirmed)."
                        + (" Head pose also leaned bowed-downward over the same stretch."
                           if pitch_mostly_down else "")
                    ),
                    "turned_away": "Head mostly turned away from the front for a sustained "
                                    "stretch (heuristic: possibly distracted -- not confirmed).",
                }[band]
                alerts.append({
                    "track_id": track_id,
                    "student_id": track["student_id"],
                    "student_name": track["student_name"],
                    "alert_type": band,
                    "detail": detail,
                    "photo_path": photo_path,
                    "severity": "high" if escalated else "medium" if band == "phone_use" else "low",
                    "needs_video_clip": escalated,
                })

        results.append({
            "track_id": track_id,
            "student_id": track["student_id"],
            "student_name": track["student_name"],
            "box": {"x": x, "y": y, "w": fw, "h": fh},
            "orientation_trend": orientation_trend,
            "band": band,
            "band_counts": dict(track["band_counts"]),
            "total_scored": track["total_scored"],
        })

    return results, alerts


BAND_COLORS = {
    "active": "#22c55e",
    "turned_away": "#f59e0b",
    "eyes_closed": "#a855f7",
    "sleeping": "#7c3aed",
    "phone_use": "#ef4444",
    "phone_use_sustained": "#b91c1c",
    "insufficient_data": "#94a3b8",
}

BAND_LABELS = {
    "active": "Active",
    "turned_away": "Turned away",
    "eyes_closed": "Eyes closed",
    "sleeping": "Sleeping (5+ min)",
    "phone_use": "Phone near face",
    "phone_use_sustained": "Phone near face (sustained)",
    "insufficient_data": "Insufficient data",
}


def boxes_for_overlay(results):
    """Same {x,y,w,h,label,color} shape the existing canvas-overlay JS
    already knows how to draw (see exam_monitor.html / old participation.html)."""
    boxes = []
    for r in results:
        box = r["box"]
        name = r["student_name"] or f"Track {r['track_id']}"
        label = f"{name} | {BAND_LABELS.get(r['band'], r['band'])}"
        boxes.append({
            "x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"],
            "label": label,
            "color": BAND_COLORS.get(r["band"], "#94a3b8"),
        })
    return boxes


def percentages(band_counts, total_scored):
    """Turn cumulative band counts into display percentages. Returns a
    dict with an 'active_pct' convenience key plus each band's own
    percentage -- always computed from total_scored (checks that had
    enough data), never from wall-clock class duration."""
    if not total_scored:
        return {"active_pct": None, "turned_away_pct": None,
                "eyes_closed_pct": None, "phone_use_pct": None}
    # "sleeping" is eyes_closed that ran long enough to escalate, and
    # phone_use_sustained is phone_use that did the same -- for the
    # session-percentage rollup they count toward their base band; the
    # escalation itself is what the alert list (not this summary) shows.
    eyes_closed_total = band_counts.get("eyes_closed", 0) + band_counts.get("sleeping", 0)
    phone_use_total = band_counts.get("phone_use", 0) + band_counts.get("phone_use_sustained", 0)
    return {
        "active_pct": round(100 * band_counts.get("active", 0) / total_scored),
        "turned_away_pct": round(100 * band_counts.get("turned_away", 0) / total_scored),
        "eyes_closed_pct": round(100 * eyes_closed_total / total_scored),
        "phone_use_pct": round(100 * phone_use_total / total_scored),
    }


# ---------------------------------------------------------------------
# Deliberately NOT implemented -- same reasoning as the module this
# replaces (participation_ai.py): true eye-contact/gaze and per-student
# speaking attribution are not reliably measurable from a classroom-
# distance camera alone. Object-detecting an actual phone (rather than
# "a hand near the face") would need a trained object detector and a
# much closer/higher-res camera than a classroom ceiling/front camera
# provides -- claiming to do it without one would be a fabricated
# signal, which this module deliberately avoids everywhere else.
# ---------------------------------------------------------------------
