# Student Attendance Management System

A simple, self-contained web app for taking daily student attendance,
built with **Python (Flask)** and **SQLite** (no external database needed).

## Features
- Admin login (secure password hashing)
- Manage classes, with schedule (start/end time, late-arrival grace period)
- Manage students (add/edit/delete), grouped by class
- Manage teachers, with their own face enrollment
- **Timetable** — define each class's daily periods (subjects, teacher, times),
  including breaks (tea/lunch) — or auto-generate a standard 8:20–4:00 day
- Mark daily attendance manually (Present / Absent / Late) per class, per date
- **📷 Live Camera attendance** — recognizes enrolled students' faces and
  marks them Present/Late for the whole day in real time
- **🎓 Smart Period Attendance** — camera first recognizes the **teacher**
  to auto-detect which period/subject is live from the timetable, then
  switches to recognizing **students** and marks attendance for that
  specific period (so DBMS 1st period and DBMS 3rd period are tracked
  separately, each against the right teacher)
- Period session history with a per-period present/absent roster
- Dashboard with today's stats and recent activity
- Attendance reports with date-range + class filters
- Export attendance report to CSV
- Change password
- **🛡️ Exam Hall Monitor (heuristic AI, USN-attributed)** — allocate
  students to a room by USN for an exam, then a room camera flags
  suspicious patterns *per candidate*: looking left/right, looking back,
  a hand lingering near the ear/face ("possible phone use"), and two
  candidates' faces unusually close together ("possible copying").
  Every alert is matched against the allocated roster and saved with
  the candidate's USN, seat number, and a proof photo. Every alert is a
  heuristic signal for a human invigilator to review, not an automatic
  verdict — see the dedicated section below.

## Requirements
- Python 3.8+
- A webcam (only needed for the camera-based attendance feature)
- `opencv-contrib-python` (installed via requirements.txt) — this is a
  larger download than a typical package since it bundles OpenCV's
  compiled binaries; give it a minute on slower connections
- The browser camera feature needs the site to be on `http://127.0.0.1`
  or `http://localhost` (which it is by default) or HTTPS — browsers
  block camera access on plain HTTP for any other address

## Setup

1. Unzip the project and open a terminal inside the folder:
   ```
   cd attendance_system
   ```

2. (Recommended) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the app:
   ```
   python app.py
   ```

5. Open your browser at: **http://127.0.0.1:5000**

The SQLite database file (`attendance.db`) is created automatically on
first run, along with a default admin account and one default class.

## Default Login
```
Username: admin
Password: admin123
```
**Change this password after first login** (use the "Change Password" link
in the navbar), especially before deploying anywhere beyond your own machine.

## Project Structure
```
attendance_system/
├── app.py                  # Main Flask application (all routes/logic)
├── exam_ai.py               # Exam Hall malpractice detection engine (heuristics)
├── download_models.py      # One-time MediaPipe model downloader
├── requirements.txt        # Python dependencies
├── attendance.db           # SQLite database (auto-created on first run)
├── face_data/               # auto-created on first face capture/train
│   ├── dataset/<student_id>/*.jpg   # enrolled face photos
│   ├── trainer.yml                 # trained LBPH recognition model
│   └── models/                     # MediaPipe model files (after download)
│       ├── face_landmarker.task
│       └── hand_landmarker.task
├── templates/              # HTML pages (Jinja2 templates)
│   ├── base.html
│   ├── login.html
│   ├── change_password.html
│   ├── dashboard.html
│   ├── classes.html
│   ├── students.html
│   ├── student_form.html
│   ├── attendance.html
│   ├── reports.html
│   ├── face_setup.html
│   ├── face_capture.html
│   ├── live_attendance.html
│   ├── exam_rooms.html             # Exam room management
│   ├── exam_room_form.html
│   ├── exam_sessions.html          # Exam Hall sessions list
│   ├── exam_session_form.html      # New exam session
│   ├── exam_allocate.html          # Seat allocation by USN
│   ├── exam_monitor.html           # Exam Hall Monitor (live camera)
│   └── exam_session_detail.html    # Exam Hall Log (history + photos)
└── static/
    ├── css/
    │   └── style.css        # Styling
    └── exam_photos/          # auto-created proof photos per exam session
```

## How to Use

### Manual attendance
1. **Login** with the default admin credentials.
2. Go to **Classes** and add your classes (e.g. "Grade 10 - A").
3. Go to **Students** → **Add Student** to add students to each class.
4. Go to **Mark Attendance**, pick a class and date, mark each student as
   Present / Absent / Late, and click **Save Attendance**. You can revisit
   any date to update it.
5. Go to **Reports** to see attendance percentages per student, filter by
   class/date range, and export to CSV for Excel.

### 🎓 Smart Period Attendance (teacher-first, per-subject tracking)
This is for a real school-day setup like: **8:20am–4:00pm, tea break at
10:20, lunch at 12:40**, with a different subject/teacher each period.

1. Go to **Teachers**, add each teacher, click **Capture Face** for each
   (same webcam flow as students), then click **Train Teacher Model**.
2. Go to **Timetable**, pick a class, and click **Generate Default Day** —
   this creates periods from 8:20 to 16:00 with the tea and lunch breaks
   already carved out. Then, for each class period, type in the **Subject**
   (e.g. "DBMS") and pick the **Teacher**, and click Save. You can also add
   periods manually if your day doesn't match the default shape.
3. Make sure both the student face model (**Face Setup** → Train Model) and
   the teacher face model are trained.
4. Go to **🎓 Smart Attendance**, pick the class, and click **Start Camera**.
   - The system checks the timetable for what's scheduled *right now*. If
     it's a break, it just shows "no attendance during breaks."
   - If it's a class period with no session yet, it looks for a **teacher's
     face**. Once it recognizes one (e.g. "Afnan"), it starts that period's
     session automatically — you'll see "Teacher: Afnan" and the subject
     (e.g. "DBMS") appear.
   - It then switches to recognizing **students** and marks each one
     Present for that specific period as they're seen, with a live
     present-count (e.g. "18 / 25").
   - Click **End Period** when the period is over (or just let the next
     period's teacher walk in — starting a new period session is separate).
5. Check **Sessions** in the navbar any time to see the full history of
   every period that ran, who taught it, and who was present — click
   **View** on any row for the full roster.

**Note:** if two different classes/rooms are running Smart Attendance at
the same time, each needs the page open with that class selected — a
single browser tab/camera tracks one class at a time, matching one
teacher walking into one room.

#### ⏰ If the teacher never shows up (auto-absent, then direct student scan)

Attendance always starts the same way: point the camera at the **teacher**
first — nothing about students is checked until a teacher's face is
recognized and matched to who the timetable says should be teaching this
period. The system waits a **30-minute grace period**
(`AUTO_ABSENT_GRACE_MINUTES` in `app.py`; a 50-minute period gives the
teacher the first 30 minutes to show up):

- **Within 30 minutes of the period's start time:** nothing happens
  automatically. Attendance stays blocked until a teacher is face-confirmed
  present, exactly as before.
- **30+ minutes after the period's start time, still no teacher recognized:**
  the period auto-opens on its own and is logged with **Teacher: Absent**.
  This happens the moment anyone views the Dashboard, Smart Attendance, or
  Live Attendance page (or the periodic background poll on those pages
  fires) — no extra setup needed. Opening the period is *all* that happens
  automatically — no student is marked present yet.
- **From that point, the camera scans for student faces directly** — every
  face in each frame is detected and matched in one pass (the same
  all-faces-at-once recognition Smart Attendance always uses), and only the
  students actually recognized get marked Present. A student who never
  appears in frame stays unmarked for that period; nobody is credited with
  attendance just because the teacher didn't show up.
- **If the teacher then walks in and gets recognized after that point:**
  they're attached to the same period as **Teacher: Late** instead of
  starting a duplicate session — any students the camera already recognized
  stay marked, and the record shows the teacher arrived late rather than
  being absent all period.

You'll see this reflected as colored badges — **Teacher Absent (auto)** /
**Teacher Late** / **Present** — on the Dashboard, Smart Attendance, and
Sessions pages. Change the grace window by editing `AUTO_ABSENT_GRACE_MINUTES`
near the top of `app.py`.

### 📷 Live Camera attendance (simpler, whole-day tracking)
This uses your webcam + OpenCV face detection/recognition — no cloud
service, everything runs on your own machine.

1. Go to **Face Setup** in the navbar.
2. For each student, click **Capture Face**. Allow camera access in your
   browser, click **Start Capture**, and look at the camera — slowly turn
   your head slightly while it automatically grabs ~25 photos of your face
   (good lighting, plain background, no sunglasses works best). Click
   **Stop** early if you want, or let it finish automatically.
3. Repeat for every student you want the system to recognize.
4. Back on the **Face Setup** page, click **Train Model**. This builds the
   recognition model from every student's captured photos (takes a few
   seconds to a minute depending on how many photos you have).
5. Go to **📷 Live Camera** in the navbar, pick the class and date, click
   **Start Camera**, and point it at students. Recognized faces get a
   green box (just marked) or blue box (already marked) drawn around them
   with their name, and the roster panel on the right updates live. An
   orange box means "face detected but not recognized."
6. You can still open **Mark Attendance** afterward to manually correct
   anyone the camera missed or misread.

**Notes on accuracy:**
- This uses OpenCV's LBPH face recognizer — good for a classroom-sized
  project, not bank-vault-grade security. Similar-looking siblings or very
  low light can occasionally be misread.
- More/better enrollment photos (different angles, consistent lighting)
  = better accuracy. Re-run **Capture Face** and **Train Model** any time
  to improve it.
- If recognition feels too strict/loose, tune
  `RECOGNITION_CONFIDENCE_THRESHOLD` near the top of `app.py` (lower =
  stricter matching, higher = more lenient).
- All face photos are stored locally in `face_data/dataset/` on your own
  machine — nothing is uploaded anywhere. Delete that folder any time to
  wipe all enrolled faces.

### 🛡️ Exam Hall Monitor (heuristic AI, USN-attributed)

This is an exam-monitoring add-on, separate from attendance. Instead of
watching a generic classroom feed, it's built around a real exam-hall
workflow: **rooms → an exam session in a room → students allocated to
that room by USN → a room camera that identifies each candidate and
flags them by name/USN**, with a saved photo as proof. Every alert is a
**heuristic** signal — a triage tool for a human invigilator, not a
verdict machine. False positives are expected (stretching, adjusting
hair, a naturally tight seating plan); nothing here is forensic-grade
proctoring.

**Signals raised, per identified candidate:**
- **Looking left / looking right** — head turned sideways, estimated
  from face landmarks (possible glancing at a neighbour's sheet).
- **Looking back** — head turned almost fully away from the desk, or a
  face-shaped region with no readable frontal landmarks at all.
- **Phone use** — a hand lingering near the ear/face area (heuristic
  proxy for "possible phone use" — this is *not* actual object
  detection of a phone, just hand position).
- **Copying** — two candidates' faces detected unusually close together
  (possible copying / passing material between neighbouring seats).
  This one works immediately with no extra setup; the other three need
  the MediaPipe model download below.

#### Setup
1. **Add exam rooms** — sidebar → **Exam Rooms** → add each hall/lab you
   run exams in, with its seating capacity.
2. **Enroll student faces** first via **Face Setup**, if you haven't —
   identification during monitoring depends on the same trained LBPH
   model used for attendance.
3. Install the new dependency: `pip install -r requirements.txt` (pulls
   in `mediapipe`, used for looking-left/right/back and phone-use).
4. Download the two small model files once:
   ```
   python download_models.py
   ```
   This needs internet access to `storage.googleapis.com`. If your
   network blocks that, the script prints the two model URLs so you can
   fetch them manually and place them in `face_data/models/`.
5. Restart the app. The Exam Hall Monitor page shows which detectors are
   active — face-matching/"copying" always are; the model-based ones
   show as "off" until the download succeeds.

#### Using it
1. Sidebar → **Exam Hall Monitor** → **+ New Exam Session** — name the
   exam, pick a room, and optionally link a class (lets you
   bulk-allocate that class's whole roster in one click).
2. On the **Seats** screen, allocate students by USN — either one at a
   time (with an optional seat number) or, if you linked a class,
   "Allocate all remaining, auto-number seats". Only allocated students
   can be identified during monitoring, since each face is checked
   against this room's roster rather than the whole school.
3. Click **Monitor** — allow camera access. Alerts stream into the
   panel on the right as they're detected, each tagged with the
   candidate's USN, seat number, severity, and a thumbnail proof photo.
4. Click **Stop** when the exam ends.
5. Every alert is saved to that session's log (sidebar → **Exam Hall
   Monitor** → **Log**), with the full timeline and proof photos you
   can review afterward, plus a per-candidate summary of who was
   flagged most.

#### Tuning
All thresholds live at the top of `exam_ai.py` — e.g.
`YAW_LOOK_THRESHOLD`, `YAW_BACK_THRESHOLD`, `NEIGHBOR_DISTANCE_FACTOR`,
`ALERT_COOLDOWN_SECONDS`. Lower thresholds = more sensitive (more
alerts, more false positives). Identification strictness is shared with
attendance recognition — `RECOGNITION_CONFIDENCE_THRESHOLD` near the top
of `app.py`.

**Notes on accuracy:** same LBPH recognizer as attendance (see notes
above) — classroom-project grade, not bank-vault-grade. Checking each
face only against the room's allocated roster (rather than the whole
school) meaningfully improves match reliability. A face that doesn't
match anyone allocated to the room is still shown ("Unidentified") and
still logged — useful as an impersonation signal, but treat it the same
as every other alert here: a pattern for a human to check, not proof.

## Notes / Customization Ideas
- This is a single-admin system by default. To support multiple teacher
  logins, add more rows to the `users` table (see `init_db()` in `app.py`)
  or build a simple "Manage Users" page following the same pattern as
  "Manage Classes".
- For production deployment, change `app.secret_key` in `app.py` to a
  long random value, and run behind a production server (e.g. gunicorn)
  instead of `python app.py`.
- The database is a single file (`attendance.db`) — back it up regularly
  if you rely on this for real records.
