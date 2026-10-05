"""
Standalone webcam test for activeness_ai.py -- no Flask, no browser.
Run this to sanity-check head-orientation / eyes-closed / phone-near-face
detection live before wiring it into the Flask app.

Usage:
    cd attendance_system
    python test_activeness.py

Press 'q' to quit.
"""

import time
import cv2

import activeness_ai

FRAME_SAMPLE_INTERVAL_SEC = 1.0  # throttle: analyze once per second, not every frame

BAND_COLORS_BGR = {
    "active": (0, 200, 0),
    "turned_away": (0, 165, 245),
    "eyes_closed": (200, 0, 160),
    "phone_use": (0, 0, 220),
    "insufficient_data": (150, 150, 150),
}


def main():
    cap = cv2.VideoCapture(0)  # change to a video file path or RTSP URL to test a classroom feed
    if not cap.isOpened():
        print("Could not open webcam (index 0). If you have an external camera, try index 1.")
        return

    state = activeness_ai.new_session_state()
    last_analyze_ts = 0.0
    last_results = []

    print("Running. Press 'q' in the video window to quit.")
    print("Reminder: every band here is a heuristic signal, not a verdict -- see activeness_ai.py.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        now = time.time()
        if now - last_analyze_ts >= FRAME_SAMPLE_INTERVAL_SEC:
            last_results, _alerts = activeness_ai.analyze_frame(frame, state)  # no identify/evidence fn yet
            last_analyze_ts = now

        for r in last_results:
            box = r["box"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            color = BAND_COLORS_BGR.get(r["band"], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            bc = r["band_counts"]
            label = (f"#{r['track_id']} {r['band']} "
                     f"(active {bc.get('active',0)}/{r['total_scored']})")
            cv2.putText(frame, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Class activeness test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
