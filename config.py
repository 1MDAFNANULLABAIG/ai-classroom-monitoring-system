"""
Centralized Configuration for AI Classroom Attendance and Exam Monitoring System
-------------------------------------------------------------------------------
All tunable parameters, model paths, and behavioral thresholds are defined here
to avoid hardcoding values across modules.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Database & Directory Paths
# ---------------------------------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
FACE_DIR = os.path.join(BASE_DIR, "face_data")
MODELS_DIR = os.path.join(FACE_DIR, "models")
DATASET_DIR = os.path.join(FACE_DIR, "dataset")
TEACHER_DATASET_DIR = os.path.join(FACE_DIR, "dataset_teachers")
TRAINER_FILE = os.path.join(FACE_DIR, "trainer.yml")
TEACHER_TRAINER_FILE = os.path.join(FACE_DIR, "trainer_teachers.yml")

# Evidence Storage Paths
STATIC_DIR = os.path.join(BASE_DIR, "static")
EXAM_PHOTOS_DIR = os.path.join(STATIC_DIR, "exam_photos")
ACTIVENESS_PHOTOS_DIR = os.path.join(STATIC_DIR, "activeness_photos")
EVIDENCE_VAULT_DIR = os.path.join(BASE_DIR, "Evidence")

# ---------------------------------------------------------------------
# Model File Paths
# ---------------------------------------------------------------------
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
FACE_LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_LANDMARKER_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_LANDMARKER_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
OBJECT_DETECTOR_PATH = os.path.join(MODELS_DIR, "efficientdet_lite0.tflite")

# ---------------------------------------------------------------------
# Enrollment Configuration
# ---------------------------------------------------------------------
SAMPLES_PER_STUDENT = 50
SAMPLES_PER_TEACHER = 50
VIDEO_ENROLLMENT_DURATION_SEC = 60
VIDEO_ENROLLMENT_MAX_EXTRACTED_FRAMES = 50
VIDEO_ENROLLMENT_MIN_SIMILARITY_DIFF = 0.08  # Cosine distance to prevent near-duplicate frames

# ---------------------------------------------------------------------
# Face Quality Thresholds
# ---------------------------------------------------------------------
MIN_FACE_SIZE = 60              # Minimum bounding box width/height in pixels
BLUR_THRESHOLD = 65.0           # Laplacian variance threshold (below = blurry)
BRIGHTNESS_MIN = 40             # Minimum average grayscale intensity (too dark)
BRIGHTNESS_MAX = 235            # Maximum average grayscale intensity (overexposed)
FACE_BOUNDARY_MARGIN = 12       # Minimum pixels from frame edge (prevent cutoff faces)
MIN_FACE_QUALITY_SCORE = 0.55   # Composite quality score threshold (0.0 to 1.0)

# ---------------------------------------------------------------------
# Face Detection & Modern Recognition
# ---------------------------------------------------------------------
FACE_DETECTION_THRESHOLD = 0.60         # YuNet detection score threshold
FACE_RECOGNITION_THRESHOLD = 0.50       # SFace cosine similarity threshold for positive match
CONFIDENCE_MARGIN_THRESHOLD = 0.08      # Safety margin: Top-1 match must beat Top-2 by at least this margin
LBPH_DISTANCE_THRESHOLD = 75            # Fallback legacy LBPH distance (lower is better)

# ---------------------------------------------------------------------
# Real-Time Multi-Frame Tracking & Verification
# ---------------------------------------------------------------------
TEMPORAL_CONFIRMATION_FRAMES = 4        # Consecutive frames required before identity is VERIFIED
TRACK_TIMEOUT = 4.0                     # Seconds to preserve track/identity during occlusion or side-angle turns
IOU_TRACK_THRESHOLD = 0.28              # Minimum IoU / spatial association threshold
RECOGNITION_INTERVAL = 3                # Run heavy embedding extraction every N frames per track
COASTING_GRACE_FRAMES = 30              # Frames to maintain tracking when face temporarily disappears

# ---------------------------------------------------------------------
# Exam Malpractice Detection & Confirmation
# ---------------------------------------------------------------------
EVENT_MIN_DURATION = 3.0                # Minimum sustained duration (seconds) required before alert
EVENT_CONFIRMATION_FRAMES = 5           # Number of consecutive/consistent frames required
EVENT_COOLDOWN = 15.0                   # Cooldown (seconds) between repeated alerts for the same behavior
CONFIDENCE_THRESHOLD = 0.75             # Default confidence threshold for behavioral classification

MALPRACTICE_DURATION_THRESHOLD = EVENT_MIN_DURATION
ALERT_COOLDOWN_SECONDS = int(EVENT_COOLDOWN)
UNKNOWN_COOLDOWN_SECONDS = 15

EVIDENCE_PRE_SECONDS = 5                # Seconds of video buffer prior to event trigger
EVIDENCE_POST_SECONDS = 5               # Seconds of video buffer following event trigger
EVIDENCE_TOTAL_CLIP_SECONDS = 10        # Target length of buffered evidence video
NEIGHBOR_COPYING_DISTANCE_FACTOR = 1.25 # Multiplier of average face/body width for proximity copying alert

# Object Detection & Seat Tracking Thresholds
PHONE_DETECTION_CONFIDENCE_MIN = 0.40   # Minimum score for cell phone / device detection
SEAT_DEVIATION_THRESHOLD_PX = 85.0      # Pixel distance threshold from seat center to flag LEAVING_SEAT
SEAT_CHANGE_DURATION_SEC = 8.0          # Seconds occupying another seat before POSSIBLE_SEAT_CHANGE
OBJECT_EXCHANGE_DISTANCE_PX = 100.0     # Maximum distance between hands of adjacent students

# Head Yaw & Pitch Thresholds (estimated from landmarks)
YAW_LOOK_THRESHOLD = 0.18               # Sideways glance -> looking_left / looking_right (approx 22 deg)
YAW_BACK_THRESHOLD = 0.40               # Extreme sideways/back -> looking_back
PITCH_DOWN_THRESHOLD = 0.22             # Head tilted down -> looking_down
PITCH_UP_THRESHOLD = -0.22              # Head tilted up -> looking_up
MOUTH_TALKING_ASPECT_RATIO = 0.38       # Mouth openness ratio for talking detection
HAND_NEAR_FACE_MARGIN = 0.55            # Bounding box multiplier for hand near face heuristic

# ---------------------------------------------------------------------
# Timetable & Grace Period Policy
# ---------------------------------------------------------------------
AUTO_ABSENT_GRACE_MINUTES = 30

# ---------------------------------------------------------------------
# Classroom Intelligence System Thresholds
# ---------------------------------------------------------------------
ATTENTION_YAW_THRESHOLD = 0.22             # Head yaw beyond which student is looking away from teaching zone
ATTENTION_PITCH_THRESHOLD = 0.22           # Head pitch beyond which student is looking down/up
SLEEP_HEAD_DOWN_DURATION = 15.0            # Seconds head down before flagging HEAD_DOWN
SLEEP_SUSTAINED_DURATION = 45.0            # Seconds continuous eyes closed/head resting before POSSIBLE_SLEEPING
EYE_CLOSED_EAR_THRESHOLD = 0.19            # Eye Aspect Ratio threshold for closed eyes
PHONE_CALL_DISTANCE_PX = 80.0              # Max pixel distance between hand/phone and ear landmark
PHONE_CONFIRMATION_DURATION = 3.0          # Seconds sustained phone usage/call before alert
HAND_RAISE_Y_OFFSET = -0.15                # Wrist Y-coordinate relative to shoulder/ear indicating raised hand
TALKING_STUDENT_DISTANCE_PX = 350.0        # Max distance between adjacent students for interaction
TALKING_MAR_THRESHOLD = 0.38               # Mouth Aspect Ratio indicating speech movement
CONFLICT_VELOCITY_THRESHOLD = 280.0        # Pixel/second landmark displacement indicating rapid aggressive movement
CONFLICT_DISTANCE_PX = 120.0               # Proximity threshold for physical conflict detection
ENGAGEMENT_WINDOW_SECONDS = 60.0           # Periodic aggregation window for classroom activity index
ID_CARD_CHEST_RATIO_MIN = 0.08             # Minimum chest area bounding box ratio to evaluate ID card
UNIFORM_COLOR_TOLERANCE = 45.0             # Delta-E color distance tolerance for uniform check
