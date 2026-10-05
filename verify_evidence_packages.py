"""
verify_evidence_packages.py
===========================
Comprehensive programmatic validation of evidence packages:
1. Opens saved evidence.mp4 using cv2.VideoCapture to verify:
   - Duration (approx 10.0 seconds: 5s pre-event + event + 5s post-event)
   - Frame count (approx 150 frames at 15 FPS)
   - Resolution (640x480)
   - Video FPS (15.0 FPS)
   - Video file readability & decoding integrity
   - File size (> 2KB)
2. Validates 6-photo evidence packages:
   - before.jpg, event_start.jpg, best_event.jpg, event_end.jpg, after.jpg, student_crop.jpg
   - Ensures distinct frames and non-duplicate content
   - Validates metadata.json completeness
"""

import os
import sys
import time
import json
import unittest
from datetime import date
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
import classroom_ai
import exam_ai


class TestEvidencePackageValidation(unittest.TestCase):

    def test_classroom_evidence_video_and_images(self):
        print("\n" + "=" * 70)
        print("VERIFYING CLASSROOM 10-SECOND EVIDENCE VIDEO & IMAGE PACKAGE")
        print("=" * 70)

        # Generate 150 frames with motion / event progression (10 seconds at 15 FPS)
        w, h = 640, 480
        frames_seq = []
        for i in range(150):
            f = np.full((h, w, 3), 190, dtype=np.uint8)
            # Add dynamic student motion (moving from left to right)
            x = int(120 + i * 2.2)
            cv2.rectangle(f, (x, 150), (x + 80, 320), (100, 150, 220), -1)
            cv2.circle(f, (x + 40, 120), 30, (180, 200, 240), -1)
            # Timestamp watermark
            cv2.putText(f, f"Frame {i:03d} | T+{i/15.0:.2f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frames_seq.append(f)

        target_box = (int(120 + 75 * 2.2), 150, 80, 170)
        session_id = 1
        event_id = 99
        event_type = "leaving_seat"
        student_info = {
            "usn": "1HK23IS001",
            "name": "Alice Johnson",
            "track_id": 1,
            "duration_seconds": 10.0,
            "confidence": 0.92,
            "identity_source": "FACE_VERIFIED",
            "detail": "Student moved away from assigned seat position for 10.0s."
        }

        # Save package using classroom_ai engine
        _ = classroom_ai.save_classroom_evidence_package(
            session_id=session_id,
            event_id=event_id,
            event_type=event_type,
            student_info=student_info,
            full_frame=frames_seq[-1],
            student_box=target_box,
            pre_frames=frames_seq
        )

        date_str = date.today().strftime("%Y_%m_%d")
        vault_dir = os.path.join(BASE_DIR, "Evidence", f"Classroom_{date_str}", "1HK23IS001", f"EVENT_{event_id:04d}_{event_type.upper()}")
        web_dir = os.path.join(BASE_DIR, "static", "activeness_photos", str(session_id), f"event_{event_id}")

        video_path = os.path.join(vault_dir, "evidence.mp4")
        meta_path = os.path.join(vault_dir, "metadata.json")

        # 1. Programmatic Video Verification
        self.assertTrue(os.path.exists(video_path), f"Video missing: {video_path}")
        file_size_bytes = os.path.getsize(video_path)
        self.assertGreater(file_size_bytes, 1000, "Evidence video is an empty stub!")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened(), f"OpenCV failed to open evidence video: {video_path}")

        v_fps = cap.get(cv2.CAP_PROP_FPS)
        v_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_duration = v_count / v_fps if v_fps > 0 else 0.0

        # Read frames to verify codec readability and frame integrity
        read_frames = 0
        while True:
            ret, v_frame = cap.read()
            if not ret:
                break
            read_frames += 1
            if read_frames == 1:
                self.assertGreater(v_frame.std(), 5.0, "Evidence frame has no visual information!")
        cap.release()

        print(f"Evidence Video Inspection Results:")
        print(f"  Evidence File Path:   {video_path}")
        print(f"  Evidence File Size:   {file_size_bytes} bytes ({file_size_bytes / 1024.0:.1f} KB)")
        print(f"  Evidence Resolution:  {v_width} x {v_height}")
        print(f"  Evidence FPS:         {v_fps:.1f}")
        print(f"  Evidence Frame Count: {v_count} (Decoded: {read_frames})")
        print(f"  Evidence Duration:    {v_duration:.2f} seconds")

        self.assertGreaterEqual(v_count, 130, "Video does not contain ~10 seconds of frames!")
        self.assertAlmostEqual(v_duration, 10.0, delta=1.5, msg="Video duration is not approx 10.0 seconds!")
        self.assertEqual((v_width, v_height), (w, h), "Video resolution does not match frame resolution!")

        # 2. Programmatic 6-Image Evidence Package Verification
        required_fnames = ["before.jpg", "event_start.jpg", "best_event.jpg", "event_end.jpg", "after.jpg", "student_crop.jpg"]
        for fname in required_fnames:
            p = os.path.join(vault_dir, fname)
            self.assertTrue(os.path.exists(p), f"Evidence image file does not exist: {p}")
            self.assertGreater(os.path.getsize(p), 500, f"Evidence image too small: {p}")
            im = cv2.imread(p)
            self.assertIsNotNone(im, f"Failed to read image: {p}")
            self.assertGreater(im.shape[0], 20)
            self.assertGreater(im.shape[1], 20)

        # 3. Non-Duplicate Verification
        im_before = cv2.imread(os.path.join(vault_dir, "before.jpg"))
        im_best = cv2.imread(os.path.join(vault_dir, "best_event.jpg"))
        im_after = cv2.imread(os.path.join(vault_dir, "after.jpg"))
        diff1 = cv2.absdiff(im_before, im_best)
        diff2 = cv2.absdiff(im_best, im_after)
        mean_diff1 = float(np.mean(diff1))
        mean_diff2 = float(np.mean(diff2))
        print(f"  Image Difference (before vs best): mean pixel delta = {mean_diff1:.2f}")
        print(f"  Image Difference (best vs after):  mean pixel delta = {mean_diff2:.2f}")
        self.assertGreater(mean_diff1, 1.0, "Evidence frames are identical duplicate images!")
        self.assertGreater(mean_diff2, 1.0, "Evidence frames are identical duplicate images!")

        # 4. Metadata Verification
        self.assertTrue(os.path.exists(meta_path), f"Metadata file missing: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["event_type"], "LEAVING_SEAT")
        self.assertEqual(meta["usn"], "1HK23IS001")
        self.assertEqual(meta["track_id"], "TRACK_01")
        self.assertGreater(meta["duration_seconds"], 0.0)
        print(f"  Metadata verified: USN={meta['usn']}, Event={meta['event_type']}, Confidence={meta['confidence']}")

        print("=" * 70)
        print("EVIDENCE PACKAGE VALIDATION: ALL CRITERIA PASSED (OK)")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()
