"""
Real-Time Multi-Person Face & Body Tracker with Temporal Verification
---------------------------------------------------------------------
Tracks students across frames using spatial IoU, centroid velocity, and seat tracking.
Implements:
  1. Multi-stage temporal identity verification:
     identity_score = face_similarity * temporal_consistency * detection_confidence * face_quality
  2. Distinct identity attribution:
     - FACE_VERIFIED: face currently visible, high confidence match
     - TRACK_ASSOCIATED: face currently invisible (turned around, backward, occluded)
       but persistent physical track maintained with grace-period coasting
     - UNKNOWN: unallocated or unverified person
  3. Seat position and departure tracking (LEAVING_SEAT, RETURNED_TO_SEAT, SEAT_CHANGE).
"""

import time
from collections import deque
import numpy as np

import config
import face_quality
import face_engine


def compute_iou(boxA, boxB):
    """Compute Intersection-over-Union between two (x, y, w, h) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    return (interArea / unionArea) if unionArea > 0 else 0.0


def compute_centroid_distance(boxA, boxB):
    """Euclidean distance between centroids of two boxes."""
    cA = (boxA[0] + boxA[2] / 2.0, boxA[1] + boxA[3] / 2.0)
    cB = (boxB[0] + boxB[2] / 2.0, boxB[1] + boxB[3] / 2.0)
    return ((cA[0] - cB[0]) ** 2 + (cA[1] - cB[1]) ** 2) ** 0.5


class TrackedPerson:
    """Represents a single physical person tracked across multiple frames."""
    def __init__(self, track_id, initial_box, timestamp):
        self.track_id = track_id
        self.box = initial_box
        self.centroid = (initial_box[0] + initial_box[2] / 2.0, initial_box[1] + initial_box[3] / 2.0)
        self.person_box = self._derive_person_box(initial_box)
        self.created_at = timestamp
        self.last_seen = timestamp
        self.coasting_frames = 0
        self.face_visible = True

        # Identity states
        self.verified_id = None
        self.verified_name = None
        self.verified_usn = None
        self.seat_no = None
        self.status = "UNDER_REVIEW"  # "VERIFIED" | "UNDER_REVIEW" | "UNKNOWN"
        self.identity_source = "UNKNOWN"  # "FACE_VERIFIED" | "TRACK_ASSOCIATED" | "UNKNOWN"
        self.last_face_verified_ts = None

        self.confidence = 0.0
        self.identity_score = 0.0
        self.face_quality_score = 0.0
        self.head_direction = "CENTER"
        self.body_orientation = "FRONT"
        self.current_activity = "NORMAL"

        # Seat / Location tracking
        self.initial_seat_box = initial_box
        self.initial_seat_centroid = self.centroid
        self.is_at_seat = True
        self.seat_leave_time = None
        self.seat_return_time = None
        self.away_duration = 0.0
        self.consecutive_away_frames = 0

        # Temporal recognition buffer: deque of prediction dicts
        self.prediction_history = deque(maxlen=10)

        # Activity & side-angle history
        self.yaw_history = deque(maxlen=15)
        self.pitch_history = deque(maxlen=15)
        self.activity_start_time = timestamp
        self.attendance_marked = False

        # Liveness tracker
        self.liveness = face_quality.LivenessTracker()

    def _derive_person_box(self, face_box):
        """Estimate upper-body/torso bounding box from face box."""
        fx, fy, fw, fh = face_box
        bx = max(0, int(fx - fw * 0.75))
        by = max(0, int(fy))
        bw = int(fw * 2.5)
        bh = int(fh * 3.5)
        return (bx, by, bw, bh)

    def update_spatial(self, box, timestamp):
        """Smooth bounding box update using exponential moving average."""
        alpha = 0.65
        self.box = (
            int(alpha * box[0] + (1 - alpha) * self.box[0]),
            int(alpha * box[1] + (1 - alpha) * self.box[1]),
            int(alpha * box[2] + (1 - alpha) * self.box[2]),
            int(alpha * box[3] + (1 - alpha) * self.box[3]),
        )
        self.centroid = (self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0)
        self.person_box = self._derive_person_box(self.box)
        self.last_seen = timestamp
        self.coasting_frames = 0
        self.face_visible = True

        self._update_seat_status(timestamp)

    def _update_seat_status(self, timestamp):
        """Checks whether the person is currently near their initial assigned seat."""
        if self.initial_seat_centroid is None:
            self.initial_seat_centroid = self.centroid
            self.initial_seat_box = self.box
            return

        dist = ((self.centroid[0] - self.initial_seat_centroid[0]) ** 2 +
                (self.centroid[1] - self.initial_seat_centroid[1]) ** 2) ** 0.5

        if dist > config.SEAT_DEVIATION_THRESHOLD_PX:
            self.consecutive_away_frames += 1
            if self.is_at_seat:
                self.is_at_seat = False
                self.seat_leave_time = timestamp
                self.seat_return_time = None
        else:
            if not self.is_at_seat:
                self.is_at_seat = True
                self.seat_return_time = timestamp
                if self.seat_leave_time:
                    self.away_duration = timestamp - self.seat_leave_time
            self.consecutive_away_frames = 0

    def add_prediction(self, match_result, quality_dict, det_score):
        """Incorporate a new recognition prediction into temporal history."""
        cand_id = match_result.get("matched_id")
        sim = match_result.get("confidence", 0.0)
        q_score = quality_dict.get("quality_score", 0.0)
        self.face_quality_score = q_score

        self.prediction_history.append({
            "candidate_id": cand_id,
            "similarity": sim,
            "quality": q_score,
            "detection": det_score,
            "status": match_result.get("status", "UNKNOWN"),
        })

        self._evaluate_temporal_identity()

    def _evaluate_temporal_identity(self):
        """
        Evaluate temporal multi-frame verification formula:
        identity_score = face_similarity * temporal_consistency * detection_confidence * face_quality
        """
        if len(self.prediction_history) < 2:
            self.status = "UNDER_REVIEW"
            self.identity_source = "UNKNOWN"
            return

        id_counts = {}
        id_sims = {}
        id_qualities = {}
        id_dets = {}

        for entry in self.prediction_history:
            cid = entry["candidate_id"]
            if cid is not None:
                id_counts[cid] = id_counts.get(cid, 0) + 1
                id_sims.setdefault(cid, []).append(entry["similarity"])
                id_qualities.setdefault(cid, []).append(entry["quality"])
                id_dets.setdefault(cid, []).append(entry["detection"])

        if not id_counts:
            self.status = "UNKNOWN"
            self.identity_source = "UNKNOWN"
            self.confidence = 0.0
            return

        best_id, count = max(id_counts.items(), key=lambda x: x[1])
        total_predictions = len(self.prediction_history)
        temporal_consistency = count / float(total_predictions)

        avg_similarity = float(np.mean(id_sims[best_id]))
        avg_quality = float(np.mean(id_qualities[best_id]))
        avg_detection = float(np.mean(id_dets[best_id]))

        score = avg_similarity * (0.5 + 0.5 * temporal_consistency) * (0.6 + 0.4 * avg_quality) * (0.7 + 0.3 * avg_detection)
        self.identity_score = round(float(score), 3)
        self.confidence = round(float(avg_similarity), 3)

        is_consistent = count >= config.TEMPORAL_CONFIRMATION_FRAMES
        is_above_thresh = avg_similarity >= config.FACE_RECOGNITION_THRESHOLD
        is_quality_pass = avg_quality >= (config.MIN_FACE_QUALITY_SCORE * 0.9)

        if is_consistent and is_above_thresh and is_quality_pass:
            self.verified_id = best_id
            self.status = "VERIFIED"
            self.identity_source = "FACE_VERIFIED"
            self.last_face_verified_ts = time.time()
        elif count >= 2 and avg_similarity >= (config.FACE_RECOGNITION_THRESHOLD * 0.85):
            self.status = "UNDER_REVIEW"
            self.identity_source = "UNKNOWN"
        else:
            self.status = "UNKNOWN"
            self.identity_source = "UNKNOWN"

    def update_pose_and_activity(self, yaw, pitch, pose, timestamp):
        """Update head direction and body orientation."""
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

        smooth_yaw = float(np.mean(self.yaw_history)) if self.yaw_history else yaw
        smooth_pitch = float(np.mean(self.pitch_history)) if self.pitch_history else pitch

        if abs(smooth_yaw) >= config.YAW_BACK_THRESHOLD:
            self.head_direction = "BACK"
            self.body_orientation = "BACK"
        elif smooth_yaw >= config.YAW_LOOK_THRESHOLD:
            self.head_direction = "LEFT"
            self.body_orientation = "LEFT" if smooth_yaw >= 0.35 else "FRONT"
        elif smooth_yaw <= -config.YAW_LOOK_THRESHOLD:
            self.head_direction = "RIGHT"
            self.body_orientation = "RIGHT" if smooth_yaw <= -0.35 else "FRONT"
        elif smooth_pitch >= config.PITCH_DOWN_THRESHOLD:
            self.head_direction = "DOWN"
        elif smooth_pitch <= config.PITCH_UP_THRESHOLD:
            self.head_direction = "UP"
        else:
            self.head_direction = "CENTER"
            self.body_orientation = "FRONT"


class ClassroomTracker:
    """Multi-person classroom & exam room tracker across consecutive camera frames."""
    def __init__(self):
        self.tracks = {}  # track_id -> TrackedPerson
        self.next_track_id = 1
        self.frame_counter = 0

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_counter = 0

    def process_frame(self, frame_bgr, gallery, roster=None, timestamp=None):
        now = timestamp or time.time()
        self.frame_counter += 1
        detections = face_engine.detect_faces(frame_bgr)

        existing_track_ids = set(self.tracks.keys())
        matched_tracks = set()
        det_matches = {}

        # Associate detections with existing tracks
        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            cost_matrix = []

            for tid in track_ids:
                track = self.tracks[tid]
                row = []
                for det in detections:
                    iou = compute_iou(track.box, det["box"])
                    cent_dist = compute_centroid_distance(track.box, det["box"])
                    avg_diag = ((track.box[2] ** 2 + track.box[3] ** 2) ** 0.5 +
                                (det["box"][2] ** 2 + det["box"][3] ** 2) ** 0.5) / 2.0

                    norm_dist = cent_dist / max(1.0, avg_diag)
                    cost = (1.0 - iou) + (norm_dist * 0.5)
                    row.append(cost)
                cost_matrix.append(row)

            used_dets = set()
            for _ in range(min(len(track_ids), len(detections))):
                min_val = float("inf")
                best_t_idx, best_d_idx = -1, -1
                for t_i, row in enumerate(cost_matrix):
                    if track_ids[t_i] in matched_tracks:
                        continue
                    for d_i, val in enumerate(row):
                        if d_i in used_dets:
                            continue
                        if val < min_val:
                            min_val = val
                            best_t_idx, best_d_idx = t_i, d_i

                cost_limit = 3.5 if len(track_ids) == 1 else 1.85
                if best_t_idx >= 0 and min_val < cost_limit:
                    tid = track_ids[best_t_idx]
                    matched_tracks.add(tid)
                    used_dets.add(best_d_idx)
                    det_matches[best_d_idx] = tid

        results = []

        # Process matched and new face detections
        for d_idx, det in enumerate(detections):
            box = det["box"]
            lms = det["landmarks"]
            score = det["score"]

            if d_idx in det_matches:
                tid = det_matches[d_idx]
                person = self.tracks[tid]
                person.update_spatial(box, now)
                if person.status == "VERIFIED":
                    person.identity_source = "FACE_VERIFIED"
                matched_tracks.add(tid)
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                person = TrackedPerson(tid, box, now)
                self.tracks[tid] = person
                matched_tracks.add(tid)

            if "is_teacher" in det:
                person.is_teacher = det["is_teacher"]
            if "velocity" in det:
                person.velocity = det["velocity"]

            # Quality assessment
            quality_info = face_quality.assess_face_quality(frame_bgr, box, lms, detection_conf=score)
            person.face_quality_score = quality_info.get("quality_score", 0.0)
            person.update_pose_and_activity(quality_info["yaw"], quality_info["pitch"], quality_info["pose"], now)
            person.liveness.update(box, lms, now)

            should_recognize = (
                person.verified_id is None or
                (self.frame_counter % config.RECOGNITION_INTERVAL == 0)
            )

            if should_recognize and gallery:
                if quality_info["quality_score"] >= (config.MIN_FACE_QUALITY_SCORE * 0.85):
                    _, emb = face_engine.extract_embedding(frame_bgr, det)
                    if emb is not None:
                        target_ids = set(roster.keys()) if roster else None
                        match_res = face_engine.match_face_against_gallery(emb, gallery, target_ids=target_ids)
                        person.add_prediction(match_res, quality_info, score)

                        if person.verified_id is not None and roster and person.verified_id in roster:
                            r_entry = roster[person.verified_id]
                            if hasattr(r_entry, "keys") and not isinstance(r_entry, dict):
                                r_entry = dict(r_entry)
                            person.verified_name = r_entry.get("name")
                            person.verified_usn = r_entry.get("roll_no") or r_entry.get("usn")
                            person.seat_no = r_entry.get("seat_no")

            results.append({
                "track_id": person.track_id,
                "label": f"Person Track #{person.track_id}",
                "student_id": person.verified_id,
                "name": person.verified_name or ("Verified" if person.status == "VERIFIED" else "Unknown"),
                "usn": person.verified_usn or "",
                "seat_no": person.seat_no or "",
                "confidence": round(person.confidence * 100.0, 1),
                "identity_score": person.identity_score,
                "face_quality": quality_info["quality_score"],
                "status": person.status,
                "identity_source": person.identity_source,
                "face_visible": True,
                "head_direction": person.head_direction,
                "body_orientation": person.body_orientation,
                "activity": person.current_activity,
                "box": person.box,
                "person_box": person.person_box,
                "is_teacher": getattr(person, "is_teacher", False),
                "velocity": getattr(person, "velocity", 0.0),
                "is_at_seat": person.is_at_seat,
                "quality_passed": quality_info["passed"],
                "quality_reasons": quality_info["reasons"],
            })

        # Process unmatched tracks that were present in previous frames
        unmatched_existing = existing_track_ids - matched_tracks
        expired = []
        for tid in unmatched_existing:
            person = self.tracks.get(tid)
            if not person:
                continue

            person.coasting_frames += 1
            person.face_visible = False
            time_lost = now - person.last_seen

            if time_lost > (config.TRACK_TIMEOUT * 2.0) or person.coasting_frames > config.COASTING_GRACE_FRAMES:
                expired.append(tid)
            else:
                if person.status == "VERIFIED":
                    person.identity_source = "TRACK_ASSOCIATED"
                else:
                    person.identity_source = "UNKNOWN"

                if person.head_direction in ("LEFT", "RIGHT", "BACK"):
                    person.body_orientation = "BACK"
                    person.head_direction = "BACK"

                person._update_seat_status(now)

                results.append({
                    "track_id": person.track_id,
                    "label": f"Person Track #{person.track_id}",
                    "student_id": person.verified_id,
                    "name": person.verified_name or "Unknown",
                    "usn": person.verified_usn or "",
                    "seat_no": person.seat_no or "",
                    "confidence": round(person.confidence * 100.0, 1),
                    "identity_score": person.identity_score,
                    "face_quality": 0.0,
                    "status": person.status,
                    "identity_source": person.identity_source,
                    "face_visible": False,
                    "head_direction": person.head_direction,
                    "body_orientation": person.body_orientation,
                    "activity": person.current_activity,
                    "box": person.box,
                    "person_box": person.person_box,
                    "is_at_seat": person.is_at_seat,
                    "quality_passed": False,
                    "quality_reasons": ["Face not currently visible (coasting/turned)"],
                })

        for tid in expired:
            self.tracks.pop(tid, None)

        return results
