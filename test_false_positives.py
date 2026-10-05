"""
test_false_positives.py
=======================
Strict False-Positive Verification Suite for Classroom & Exam Monitoring:
1. Normal writing -> No sleeping/head-down alert
2. Normal reading -> No inattention alert
3. Looking down at notebook -> No sleeping alert
4. Brief left glance (< 1.5s) -> No cheating alert
5. Brief right glance (< 1.5s) -> No cheating alert
6. Phone-like object without hand/ear proximity -> No false phone alert
7. Students sitting together facing forward -> No talking/interaction alert
8. Teacher interacting with student -> Not classified as student misconduct
9. Normal postural movement (< 85px) -> No leaving seat alert
10. Occluded lower body -> Reports SHOES_NOT_VISIBLE, never false compliance
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


class TestFalsePositives(unittest.TestCase):

    def setUp(self):
        self.img = np.full((480, 640, 3), 210, dtype=np.uint8)

    def _mock_face_landmarks(self, nose_x=220, nose_y=200, eye_span=80, chin_y=240, mouth_open=0.0):
        mf_pts = np.zeros((468, 2), dtype=np.float32)
        half_span = eye_span / 2.0
        mf_pts[1] = [nose_x, nose_y]
        mf_pts[33] = [220 - half_span, 190]
        mf_pts[263] = [220 + half_span, 190]
        mf_pts[152] = [220, chin_y]
        mf_pts[13] = [220, 215]
        mf_pts[14] = [220, 215 + mouth_open]
        mf_pts[78] = [200, 215]
        mf_pts[308] = [240, 215]

        mock_fl = MagicMock()
        mock_lm = [MagicMock(x=pt[0] / 640.0, y=pt[1] / 480.0) for pt in mf_pts]
        mock_fl.face_landmarks = [mock_lm]
        return mock_fl

    # 1. Normal writing -> no sleeping alert
    def test_01_normal_writing_no_sleeping_alert(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        now = time.time()
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), now)
        p.verified_usn = "1HK23IS001"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        mock_fl = self._mock_face_landmarks(nose_x=220, nose_y=222, chin_y=255)  # slight downward pitch for writing

        with patch("face_engine.detect_faces", return_value=det), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.0, "pitch": 0.15, "pose": "FRONT", "quality_score": 0.85, "passed": True, "reasons": []}), \
             patch.object(classroom_ai, "_face_landmarker", create=True) as fl_mock:
            fl_mock.detect.return_value = mock_fl

            for _ in range(5):
                res = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st = res["students"][0]
            self.assertIn(st["current_activity"], ("WRITING", "READING", "ATTENTIVE"))
            self.assertNotIn("SLEEP", st["current_activity"])
            sleep_alerts = [a for a in res["alerts"] if "sleep" in a.get("type", "").lower()]
            self.assertEqual(len(sleep_alerts), 0, "False positive sleeping alert triggered during normal writing!")
            print("  [OK] Normal writing: No sleeping alert triggered.")

    # 2. Normal reading -> no inattention alert
    def test_02_normal_reading_no_inattention_alert(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), time.time())
        p.verified_usn = "1HK23IS002"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        mock_fl = self._mock_face_landmarks(nose_x=220, nose_y=218, mouth_open=5.0)

        with patch("face_engine.detect_faces", return_value=det), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.0, "pitch": 0.12, "pose": "FRONT", "quality_score": 0.85, "passed": True, "reasons": []}), \
             patch.object(classroom_ai, "_face_landmarker", create=True) as fl_mock:
            fl_mock.detect.return_value = mock_fl

            for _ in range(5):
                res = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st = res["students"][0]
            self.assertIn(st["current_activity"], ("READING", "WRITING", "ATTENTIVE"))
            self.assertNotEqual(st["attention_state"], "LOOKING_AWAY")
            print("  [OK] Normal reading: Classified as READING without inattention alert.")

    # 3. Looking down at notebook -> no sleeping alert
    def test_03_looking_down_notebook_no_sleeping_alert(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), time.time())
        p.verified_usn = "1HK23IS003"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        mock_fl = self._mock_face_landmarks(nose_x=220, nose_y=225)

        with patch("face_engine.detect_faces", return_value=det), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.0, "pitch": 0.18, "pose": "FRONT", "quality_score": 0.82, "passed": True, "reasons": []}), \
             patch.object(classroom_ai, "_face_landmarker", create=True) as fl_mock:
            fl_mock.detect.return_value = mock_fl

            res = classroom_ai.analyze_frame(self.img, state, session_id=1)
            sleep_alerts = [a for a in res["alerts"] if "sleep" in a.get("type", "").lower()]
            self.assertEqual(len(sleep_alerts), 0)
            print("  [OK] Looking down at notebook: No false sleeping alert.")

    # 4. Brief left glance (< 1.5s) -> no cheating alert
    def test_04_brief_left_glance_no_cheating_alert(self):
        state = exam_ai.new_session_state()
        trk = state["tracker"]
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), time.time())
        p.verified_usn = "1HK23IS004"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        # Brief glance for only 2 frames (approx 0.1s, far below 2.0s threshold)
        with patch("face_engine.detect_faces", return_value=det), \
             patch("face_quality.assess_face_quality", return_value={"yaw": -0.28, "pitch": 0.0, "pose": "LEFT", "quality_score": 0.85, "passed": True, "reasons": []}):
            for _ in range(2):
                alerts, _, _ = exam_ai.analyze_frame(self.img, state, session_id=1)

            left_alerts = [a for a in alerts if a.get("type") in ("looking_left", "possible_copying_look")]
            self.assertEqual(len(left_alerts), 0, "Brief glance (< 1.5s) triggered premature cheating alert!")
            print("  [OK] Brief left glance: Temporal persistence filter prevented false alert.")

    # 5. Brief right glance (< 1.5s) -> no cheating alert
    def test_05_brief_right_glance_no_cheating_alert(self):
        state = exam_ai.new_session_state()
        trk = state["tracker"]
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), time.time())
        p.verified_usn = "1HK23IS005"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        with patch("face_engine.detect_faces", return_value=det), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.28, "pitch": 0.0, "pose": "RIGHT", "quality_score": 0.85, "passed": True, "reasons": []}):
            for _ in range(2):
                alerts, _, _ = exam_ai.analyze_frame(self.img, state, session_id=1)

            right_alerts = [a for a in alerts if a.get("type") in ("looking_right", "possible_copying_look")]
            self.assertEqual(len(right_alerts), 0, "Brief right glance (< 1.5s) triggered premature cheating alert!")
            print("  [OK] Brief right glance: No false cheating alert.")

    # 6. Uncertain phone-like object -> no phone alert without confirmation
    def test_06_uncertain_phone_no_false_alert(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), time.time())
        p.verified_usn = "1HK23IS006"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        # Mock object detector returning a phone far outside student zone or 1 single frame
        mock_obj = MagicMock()
        mock_obj.category_name = "cell phone"
        mock_obj.score = 0.35  # low confidence
        mock_obj.bounding_box = MagicMock(origin_x=10, origin_y=10, width=30, height=50)

        with patch("face_engine.detect_faces", return_value=det), \
             patch.object(classroom_ai, "_object_detector", create=True) as od_mock:
            od_mock.detect.return_value = MagicMock(detections=[mock_obj])

            res = classroom_ai.analyze_frame(self.img, state, session_id=1)
            phone_alerts = [a for a in res["alerts"] if "phone" in a.get("type", "").lower()]
            self.assertEqual(len(phone_alerts), 0, "Uncertain/distant phone triggered false alert!")
            print("  [OK] Uncertain phone object: Ignored when low confidence / not in student zone.")

    # 7. Students sitting together facing forward -> no talking alert
    def test_07_students_sitting_together_facing_forward(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        now = time.time()
        p1 = tracker.TrackedPerson(1, (150, 150, 80, 80), now)
        p1.verified_usn = "1HK23IS007"
        p1.status = "VERIFIED"
        p2 = tracker.TrackedPerson(2, (280, 150, 80, 80), now)
        p2.verified_usn = "1HK23IS008"
        p2.status = "VERIFIED"
        trk.tracks[1] = p1
        trk.tracks[2] = p2

        det = [
            {"box": (150, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None},
            {"box": (280, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None},
        ]
        # Forward facing, mouth closed
        mock_fl = self._mock_face_landmarks(nose_x=220, nose_y=200, mouth_open=0.0)

        with patch("face_engine.detect_faces", return_value=det), \
             patch("face_quality.assess_face_quality", return_value={"yaw": 0.0, "pitch": 0.0, "pose": "FRONT", "quality_score": 0.9, "passed": True, "reasons": []}), \
             patch.object(classroom_ai, "_face_landmarker", create=True) as fl_mock:
            fl_mock.detect.return_value = mock_fl

            for _ in range(5):
                res = classroom_ai.analyze_frame(self.img, state, session_id=1)

            talk_alerts = [a for a in res["alerts"] if "talk" in a.get("type", "").lower()]
            self.assertEqual(len(talk_alerts), 0, "Adjacent quiet students triggered false talking alert!")
            print("  [OK] Adjacent students facing front: No false talking alert.")

    # 8. Teacher interaction -> not student misconduct
    def test_08_teacher_interaction_not_misconduct(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        now = time.time()
        # Student
        p_st = tracker.TrackedPerson(1, (150, 150, 80, 80), now)
        p_st.verified_usn = "1HK23IS009"
        p_st.status = "VERIFIED"
        p_st.is_teacher = False
        trk.tracks[1] = p_st
        # Teacher standing near student
        p_tc = tracker.TrackedPerson(2, (200, 130, 90, 90), now)
        p_tc.verified_name = "Prof. Smith"
        p_tc.status = "VERIFIED"
        p_tc.is_teacher = True
        trk.tracks[2] = p_tc

        det = [
            {"box": (150, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None, "is_teacher": False},
            {"box": (200, 130, 90, 90), "landmarks": None, "score": 0.95, "raw_face": None, "is_teacher": True},
        ]

        with patch("face_engine.detect_faces", return_value=det):
            res = classroom_ai.analyze_frame(self.img, state, session_id=1)

            # Ensure teacher interaction is classified properly, not as physical conflict or misconduct
            conflict_alerts = [a for a in res["alerts"] if "conflict" in a.get("type", "").lower()]
            self.assertEqual(len(conflict_alerts), 0, "Teacher near student falsely flagged as physical conflict!")
            self.assertEqual(res["summary"]["teacher_activity"], "TEACHER_INTERACTION")
            print("  [OK] Teacher interacting with student: Classified as TEACHER_INTERACTION, zero misconduct alerts.")

    # 9. Normal small postural movement -> no leaving seat alert
    def test_09_normal_movement_no_leaving_seat(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        now = time.time()
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), now)
        p.verified_usn = "1HK23IS010"
        p.status = "VERIFIED"
        p.assigned_seat_box = (200, 150, 80, 80)
        trk.tracks[1] = p

        # Shift by only 15px (well below 85px seat departure threshold)
        det = [{"box": (215, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]

        with patch("face_engine.detect_faces", return_value=det):
            for _ in range(5):
                res = classroom_ai.analyze_frame(self.img, state, session_id=1)

            st = res["students"][0]
            self.assertEqual(st["seat_status"], "NORMAL")
            seat_alerts = [a for a in res["alerts"] if "seat" in a.get("type", "").lower()]
            self.assertEqual(len(seat_alerts), 0, "Small postural shift triggered false leaving seat alert!")
            print("  [OK] Normal small postural shift (15px): Remains seat_status = NORMAL.")

    # 10. Lower body occluded -> Honest SHOES_NOT_VISIBLE
    def test_10_shoes_occluded_honest_reporting(self):
        state = classroom_ai.new_session_state()
        trk = state["tracker"]
        p = tracker.TrackedPerson(1, (200, 150, 80, 80), time.time())
        p.verified_usn = "1HK23IS011"
        p.status = "VERIFIED"
        trk.tracks[1] = p

        det = [{"box": (200, 150, 80, 80), "landmarks": None, "score": 0.95, "raw_face": None}]
        with patch("face_engine.detect_faces", return_value=det):
            res = classroom_ai.analyze_frame(self.img, state, session_id=1)
            st = res["students"][0]
            self.assertEqual(st["shoes"], "SHOES_NOT_VISIBLE")
            self.assertNotEqual(st["shoes"], "COMPLIANT")
            print("  [OK] Occluded lower body: Honestly reports SHOES_NOT_VISIBLE, zero false compliance.")


if __name__ == "__main__":
    unittest.main()
