"""
Modern Face Detection, Alignment, SFace Embedding and Multi-Gallery Matching Engine
-----------------------------------------------------------------------------------
Integrates OpenCV YuNet (5-point detection), SFace (128-D aligned embeddings),
multi-embedding gallery comparison with top-1/top-2 margin checks, side-angle
handling, 60-second video keyframe extraction, and backward-compatible LBPH hybrid.
"""

import os
import glob
import time
import sqlite3
import numpy as np
import cv2

import config
import face_quality

# ---------------------------------------------------------------------
# Global Models & Caches
# ---------------------------------------------------------------------
_detector_yn = None
_recognizer_sf = None
_detector_input_size = (320, 320)

# In-memory embedding galleries:
# student_gallery = { student_id: [ {"embedding": np.ndarray (128,), "pose": str, "quality": float} ] }
_student_gallery = {}
_teacher_gallery = {}

# Legacy LBPH recognizers (for backward compatibility)
_lbph_student = None
_lbph_teacher = None


def get_detector():
    """Lazily load OpenCV YuNet Face Detector."""
    global _detector_yn, _detector_input_size
    if _detector_yn is None:
        if os.path.exists(config.YUNET_MODEL_PATH):
            try:
                _detector_yn = cv2.FaceDetectorYN.create(
                    config.YUNET_MODEL_PATH,
                    "",
                    _detector_input_size,
                    score_threshold=config.FACE_DETECTION_THRESHOLD,
                    nms_threshold=0.3,
                    top_k=50,
                )
            except Exception as e:
                print(f"[warning] Failed to load YuNet detector: {e}")
                _detector_yn = None
    return _detector_yn


def get_recognizer_sf():
    """Lazily load OpenCV SFace Feature Recognizer."""
    global _recognizer_sf
    if _recognizer_sf is None:
        if os.path.exists(config.SFACE_MODEL_PATH):
            try:
                _recognizer_sf = cv2.FaceRecognizerSF.create(config.SFACE_MODEL_PATH, "")
            except Exception as e:
                print(f"[warning] Failed to load SFace recognizer: {e}")
                _recognizer_sf = None
    return _recognizer_sf


def detect_faces(image_bgr):
    """
    Detect faces in an image using YuNet (or fallback Haar cascade).
    Returns list of dicts:
      [
        {
          "box": (x, y, w, h),
          "landmarks": np.ndarray (5, 2), # re, le, nt, rc, lc
          "score": float,
          "raw_face": np.ndarray (15,) # raw YuNet face row
        }, ...
      ]
    """
    h, w = image_bgr.shape[:2]
    detector = get_detector()

    if detector is not None:
        try:
            # Update input size if frame dimension changed
            detector.setInputSize((w, h))
            _, faces = detector.detect(image_bgr)
            results = []
            if faces is not None:
                for face in faces:
                    bx, by, bw, bh = face[0:4].astype(int)
                    # YuNet landmarks: 5 points [re, le, nt, rm, lm]
                    lms = face[4:14].reshape((5, 2)).astype(np.float32)
                    score = float(face[14])
                    results.append({
                        "box": (int(bx), int(by), int(bw), int(bh)),
                        "landmarks": lms,
                        "score": score,
                        "raw_face": face,
                    })
                return results
        except Exception as e:
            pass

    # Fallback to Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    haar_boxes = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(config.MIN_FACE_SIZE, config.MIN_FACE_SIZE))
    results = []
    for (bx, by, bw, bh) in haar_boxes:
        results.append({
            "box": (int(bx), int(by), int(bw), int(bh)),
            "landmarks": None,
            "score": 0.85,
            "raw_face": None,
        })
    return results


def extract_embedding(image_bgr, face_det):
    """
    Align face and extract 128-D L2-normalized feature embedding using SFace.
    face_det: dict from detect_faces()
    Returns: (aligned_crop, embedding_128d_or_None)
    """
    sface = get_recognizer_sf()
    if sface is None:
        return None, None

    raw_face = face_det.get("raw_face")
    if raw_face is not None:
        try:
            aligned_face = sface.alignCrop(image_bgr, raw_face)
            embedding = sface.feature(aligned_face)
            # Ensure shape is (128,) float32
            embedding = embedding.flatten().astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            return aligned_face, embedding
        except Exception:
            pass

    # Fallback alignment using box
    x, y, w, h = face_det["box"]
    fh, fw = image_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size > 0:
        crop_resized = cv2.resize(crop, (112, 112))
        try:
            embedding = sface.feature(crop_resized)
            embedding = embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            return crop_resized, embedding
        except Exception:
            pass

    return None, None


def compute_cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two 128-D normalized embeddings. Range: -1.0 to +1.0."""
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2))


# ---------------------------------------------------------------------
# Gallery Management & Multi-Angle Embedding Store
# ---------------------------------------------------------------------
def load_gallery(is_teacher=False, force_reload=False):
    """
    Load face embeddings gallery for all registered students or teachers.
    Extracts embeddings from dataset folders and caches them in memory.
    """
    global _student_gallery, _teacher_gallery
    gallery = _teacher_gallery if is_teacher else _student_gallery
    dataset_dir = config.TEACHER_DATASET_DIR if is_teacher else config.DATASET_DIR

    if gallery and not force_reload:
        return gallery

    gallery.clear()
    if not os.path.isdir(dataset_dir):
        return gallery

    sface = get_recognizer_sf()
    detector = get_detector()

    for entry in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(person_dir):
            continue
        try:
            person_id = int(entry)
        except ValueError:
            continue

        img_paths = glob.glob(os.path.join(person_dir, "*.jpg"))
        person_embeddings = []

        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                continue

            # Run detection & embedding
            faces = detect_faces(img)
            if faces:
                best_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                quality = face_quality.assess_face_quality(img, best_face["box"], best_face["landmarks"])
                _, emb = extract_embedding(img, best_face)
                if emb is not None:
                    person_embeddings.append({
                        "embedding": emb,
                        "pose": quality["pose"],
                        "quality": quality["quality_score"],
                    })

        if person_embeddings:
            gallery[person_id] = person_embeddings

    return gallery


def match_face_against_gallery(embedding, gallery, target_ids=None):
    """
    Match an embedding against a gallery with top-1 vs top-2 margin check.
    target_ids: optional set/list of allowed person_ids (e.g. room roster).
    Returns:
      {
         "matched_id": int or None,
         "confidence": float (0.0 to 1.0),
         "top_matches": [ (person_id, score), ... ],
         "margin": float,
         "status": "VERIFIED" | "UNKNOWN" | "AMBIGUOUS"
      }
    """
    if embedding is None or not gallery:
        return {"matched_id": None, "confidence": 0.0, "top_matches": [], "margin": 0.0, "status": "UNKNOWN"}

    scores = {}
    for person_id, templates in gallery.items():
        if target_ids is not None and person_id not in target_ids:
            continue

        # Compute max cosine similarity across all stored templates for this person
        sims = [compute_cosine_similarity(embedding, t["embedding"]) for t in templates]
        if sims:
            # Top-1 max similarity + slight bonus for consistent multi-template agreement
            sims.sort(reverse=True)
            best_sim = sims[0]
            # If multiple templates match well, take smoothed average of top 3
            top3 = sims[:min(3, len(sims))]
            smoothed_sim = float(np.mean(top3))
            scores[person_id] = max(best_sim, smoothed_sim)

    if not scores:
        return {"matched_id": None, "confidence": 0.0, "top_matches": [], "margin": 0.0, "status": "UNKNOWN"}

    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top1_id, top1_sim = sorted_candidates[0]
    top2_sim = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
    margin = top1_sim - top2_sim

    # Margin check & confidence threshold
    if top1_sim < config.FACE_RECOGNITION_THRESHOLD:
        return {
            "matched_id": None,
            "confidence": round(top1_sim, 3),
            "top_matches": sorted_candidates[:3],
            "margin": round(margin, 3),
            "status": "UNKNOWN",
        }

    if len(sorted_candidates) > 1 and margin < config.CONFIDENCE_MARGIN_THRESHOLD:
        # Ambiguous match between two very similar candidates
        return {
            "matched_id": None,
            "confidence": round(top1_sim, 3),
            "top_matches": sorted_candidates[:3],
            "margin": round(margin, 3),
            "status": "AMBIGUOUS",
        }

    return {
        "matched_id": top1_id,
        "confidence": round(top1_sim, 3),
        "top_matches": sorted_candidates[:3],
        "margin": round(margin, 3),
        "status": "VERIFIED",
    }


# ---------------------------------------------------------------------
# 60-Second Video Enrollment Keyframe Extractor (Item 2)
# ---------------------------------------------------------------------
def extract_diverse_enrollment_frames_from_video(video_path, target_count=config.VIDEO_ENROLLMENT_MAX_EXTRACTED_FRAMES):
    """
    Extracts approximately 50 high-quality, pose-diverse face crops from a 60s enrollment video.
    Filters out blurry/small frames and duplicate poses.
    Returns: list of (frame_bgr, box, quality_dict, embedding)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_interval = max(1, int(fps * 0.4))  # Sample roughly 2.5 times per second

    candidates = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            faces = detect_faces(frame)
            if len(faces) == 1:
                face = faces[0]
                q = face_quality.assess_face_quality(frame, face["box"], face["landmarks"])
                if q["passed"]:
                    _, emb = extract_embedding(frame, face)
                    if emb is not None:
                        candidates.append({
                            "frame": frame.copy(),
                            "box": face["box"],
                            "quality": q,
                            "embedding": emb,
                            "pose": q["pose"],
                        })
        frame_idx += 1

    cap.release()

    if not candidates:
        return []

    # Sort candidates into pose buckets:
    # FRONT, LEFT_SLIGHT, LEFT_MEDIUM, LEFT_LARGE, RIGHT_SLIGHT, RIGHT_MEDIUM, RIGHT_LARGE, UP, DOWN
    pose_buckets = {}
    for c in candidates:
        pose_buckets.setdefault(c["pose"], []).append(c)

    selected = []
    selected_embeddings = []

    # Interleave selection across poses to ensure balanced diversity
    poses = list(pose_buckets.keys())
    # Sort each bucket by highest quality first
    for p in poses:
        pose_buckets[p].sort(key=lambda x: x["quality"]["quality_score"], reverse=True)

    idx = 0
    while len(selected) < target_count and any(pose_buckets.values()):
        pose = poses[idx % len(poses)]
        idx += 1
        if not pose_buckets[pose]:
            continue

        cand = pose_buckets[pose].pop(0)

        # Duplicate check: ensure candidate is sufficiently distinct from already chosen frames
        is_dup = False
        for chosen_emb in selected_embeddings:
            sim = compute_cosine_similarity(cand["embedding"], chosen_emb)
            if sim > (1.0 - config.VIDEO_ENROLLMENT_MIN_SIMILARITY_DIFF):
                is_dup = True
                break

        if not is_dup:
            selected.append((cand["frame"], cand["box"], cand["quality"], cand["embedding"]))
            selected_embeddings.append(cand["embedding"])

    # If still under target, fill with highest remaining quality frames
    remaining = [c for bucket in pose_buckets.values() for c in bucket]
    remaining.sort(key=lambda x: x["quality"]["quality_score"], reverse=True)
    for cand in remaining:
        if len(selected) >= target_count:
            break
        selected.append((cand["frame"], cand["box"], cand["quality"], cand["embedding"]))

    return selected
