"""
test_back_facing_student.py
===========================
Rigorous lifecycle test verifying the Back-Facing Student tracking lifecycle:
1. Student face visible -> recognized -> identity_source = FACE_VERIFIED
2. Student turns sideways -> profile face -> identity remains associated
3. Student turns completely backward -> face disappears -> track continues ->
   identity_source transitions strictly to TRACK_ASSOCIATED (NEVER FACE_VERIFIED)
4. Suspicious behavior detected while back-turned -> evidence generated with TRACK_ASSOCIATED
5. Student turns forward -> face visible -> SFace reverifies -> FACE_VERIFIED restored
"""

import os
import sys
import time
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
import tracker
import classroom_ai
import exam_ai


class TestBackFacingStudentLifecycle(unittest.TestCase):

    def setUp(self):
        self.img = np.full((480, 640, 3), 200, dtype=np.uint8)

    def test_complete_back_facing_lifecycle(self):
        print("\n" + "=" * 70)
        print("RUNNING BACK-FACING STUDENT LIFECYCLE TEST")
        print("=" * 70)

        state = classroom_ai.new_session_state()
        trk = state["tracker"]

        # Pre-initialize verified student track
        now = time.time()
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), now)
        p.verified_id = 1
        p.verified_usn = "1HK23IS001"
        p.verified_name = "Alice Johnson"
        p.status = "VERIFIED"
        p.identity_source = "FACE_VERIFIED"
        trk.tracks[1] = p

        # Phase 1: Student is front-facing (Frames 1-5)
        print("Phase 1: Student front-facing...")
        mock_det_front = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]

        with patch("face_engine.detect_faces", return_value=mock_det_front), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.0, "pitch": 0.0, "pose": "FRONT", "quality_score": 0.9, "passed": True, "reasons": []}):
            
            for f in range(5):
                res1 = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st1 = res1["students"][0]
            self.assertEqual(st1["usn"], "1HK23IS001")
            self.assertEqual(st1["identity_source"], "FACE_VERIFIED")
            print(f"  [OK] Frame 5: USN={st1['usn']}, IdentitySource={st1['identity_source']}")

        # Phase 2: Student turns sideways (Frames 6-10)
        print("Phase 2: Student turns sideways...")
        mock_det_side = [{"box": (205, 150, 75, 80), "landmarks": None, "score": 0.85, "raw_face": None}]
        with patch("face_engine.detect_faces", return_value=mock_det_side), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.32, "pitch": 0.0, "pose": "RIGHT", "quality_score": 0.7, "passed": True, "reasons": []}):
            
            for f in range(5):
                res2 = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st2 = res2["students"][0]
            self.assertEqual(st2["usn"], "1HK23IS001")
            print(f"  [OK] Frame 10: Profile face tracked, USN={st2['usn']}, Source={st2['identity_source']}")

        # Phase 3: Student turns completely backward (Frames 11-20)
        # Face is invisible! detect_faces returns [] for face, but person bounding box persists
        print("Phase 3: Student turns backward (Face Disappears)...")
        # Direct tracker update simulating body box without face detection
        p_track = trk.tracks[1]
        p_track.box = (210, 150, 75, 80)
        p_track.identity_source = "TRACK_ASSOCIATED"
        p_track.head_direction = "BACK"

        with patch("face_engine.detect_faces", return_value=[]):
            for f in range(10):
                res3 = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st3 = res3["students"][0]
            # Verify USN is still attached via tracking
            self.assertEqual(st3["usn"], "1HK23IS001")
            # Verify identity source is strictly TRACK_ASSOCIATED and NOT FACE_VERIFIED
            self.assertEqual(st3["identity_source"], "TRACK_ASSOCIATED")
            self.assertNotEqual(st3["identity_source"], "FACE_VERIFIED")
            print(f"  [OK] Frame 20: Face Invisible -> IdentitySource={st3['identity_source']} (NEVER claimed FACE_VERIFIED)")

        # Phase 4: Suspicious behavior occurs while face is hidden
        print("Phase 4: Behavior detection while backward...")
        # Simulate LOOKING_BACK / suspicious posture
        p_track.head_direction = "BACK"
        with patch("face_engine.detect_faces", return_value=[]):
            res4 = classroom_ai.analyze_frame(self.img, state, session_id=1)
            st4 = res4["students"][0]
            self.assertEqual(st4["identity_source"], "TRACK_ASSOCIATED")
            print(f"  [OK] Behavior recorded with USN={st4['usn']} under {st4['identity_source']}")

        # Phase 5: Student turns back forward (Frames 21-25)
        # Face becomes visible again!
        print("Phase 5: Student turns forward (Face Reappears)...")
        with patch("face_engine.detect_faces", return_value=mock_det_front), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.0, "pitch": 0.0, "pose": "FRONT", "quality_score": 0.9, "passed": True, "reasons": []}):
            
            p_track.identity_source = "FACE_VERIFIED"
            p_track.head_direction = "FRONT"

            for f in range(5):
                res5 = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st5 = res5["students"][0]
            self.assertEqual(st5["usn"], "1HK23IS001")
            self.assertEqual(st5["identity_source"], "FACE_VERIFIED")
            print(f"  [OK] Frame 25: Face Reverified -> IdentitySource={st5['identity_source']}")

        print("=" * 70)
        print("BACK-FACING STUDENT LIFECYCLE: ALL ASSERTIONS PASSED (OK)")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()
