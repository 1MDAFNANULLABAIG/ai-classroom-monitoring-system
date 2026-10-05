"""
Model Downloader for AI Classroom Attendance and Exam Monitoring System
------------------------------------------------------------------------
Downloads the 6 required vision models:
  1. MediaPipe FaceLandmarker (3D mesh, yaw/pitch, EAR, MAR)
  2. MediaPipe HandLandmarker (phone/hand near face)
  3. MediaPipe PoseLandmarker (33 3D full-body pose points, body orientation)
  4. MediaPipe ObjectDetector (EfficientDet Lite0: cell phone, book, laptop, etc.)
  5. OpenCV YuNet (fast 5-point face detection & alignment)
  6. OpenCV SFace (128-D normalized face embedding extractor)

Usage:
  python download_models.py
"""

import os
import shutil
import urllib.request
import ssl

import config

MODELS_DIR = config.MODELS_DIR

FILES = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite",
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download_file(url, dest):
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Checking and downloading vision models...\n")

    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[skip] {filename} already exists ({os.path.getsize(dest):,} bytes)")
            continue

        print(f"[download] {filename} ...")
        try:
            download_file(url, dest)
            print(f"[ok] saved to {dest} ({os.path.getsize(dest):,} bytes)")
        except Exception as e:
            print(f"[FAILED] {filename}: {e}")
            print(f"         Manual download URL: {url}")

    print("\nAll models ready. Restart the application to load them.")


if __name__ == "__main__":
    main()
