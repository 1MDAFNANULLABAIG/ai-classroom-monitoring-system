"""
Automatic Data Expiry (Advanced AI Feature 9.4)
--------------------------------------------------------------------
Privacy / storage-optimization helper. Identifies students whose
FACE PHOTOS should be deleted, based on two independent triggers --
either one is enough:

  1. Graduation / inactive status -- a student marked 'graduated' or
     'inactive' on the Students page, once a grace period has passed
     (GRADUATED_GRACE_DAYS). This is "graduation detection" done
     honestly: a human marks the status (there's no reliable signal
     in this app that would let software infer someone has graduated
     on its own), and the system handles the grace period + deletion.

  2. Long inactivity -- no attendance activity at all for
     INACTIVITY_DAYS, regardless of status. Catches students nobody
     remembered to mark graduated/inactive.

Deliberately scoped to FACE PHOTOS ONLY. Attendance history,
discipline records, and the student row itself are academic/
administrative records and are left alone -- this module never
deletes those.

This module only *identifies candidates* (find_due_for_expiry) --
it never touches the filesystem or the database itself. app.py
decides what to do with the list: purge immediately (background
sweep) or show it to an admin first (manual preview page), and is
responsible for actually deleting files and writing the audit log.
"""

from datetime import date, datetime

GRADUATED_GRACE_DAYS = 30   # days after being marked graduated/inactive
INACTIVITY_DAYS = 365       # days of zero attendance activity


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:10]).date()
    except ValueError:
        return None


def get_last_activity_date(conn, student_id):
    """Most recent date this student has an attendance record at all."""
    row = conn.execute(
        "SELECT MAX(att_date) AS d FROM attendance WHERE student_id = ?",
        (student_id,),
    ).fetchone()
    return _parse_date(row["d"]) if row else None


def find_due_for_expiry(conn, has_face_data_fn, today=None):
    """
    Returns a list of candidates whose face photos should be purged.
    has_face_data_fn(student_id) -> bool is injected so this module
    stays filesystem-free (easy to reason about and test).
    """
    today = today or date.today()
    students = conn.execute(
        "SELECT s.id, s.roll_no, s.name, s.status, s.status_set_at, c.name AS class_name "
        "FROM students s JOIN classes c ON c.id = s.class_id"
    ).fetchall()

    due = []
    for s in students:
        if not has_face_data_fn(s["id"]):
            continue  # nothing to delete

        reason, detail = None, None

        if s["status"] in ("graduated", "inactive") and s["status_set_at"]:
            set_at = _parse_date(s["status_set_at"])
            if set_at:
                age_days = (today - set_at).days
                if age_days >= GRADUATED_GRACE_DAYS:
                    reason = "status"
                    detail = (
                        f"Marked '{s['status']}' on {set_at.isoformat()} "
                        f"({age_days} days ago, grace period is {GRADUATED_GRACE_DAYS} days)."
                    )

        if reason is None:
            last_active = get_last_activity_date(conn, s["id"])
            if last_active:
                age_days = (today - last_active).days
                if age_days >= INACTIVITY_DAYS:
                    reason = "inactivity"
                    detail = (
                        f"No attendance activity since {last_active.isoformat()} "
                        f"({age_days} days ago, threshold is {INACTIVITY_DAYS} days)."
                    )

        if reason:
            due.append({
                "student_id": s["id"],
                "roll_no": s["roll_no"],
                "name": s["name"],
                "class_name": s["class_name"],
                "status": s["status"],
                "reason": reason,
                "detail": detail,
            })

    return due
