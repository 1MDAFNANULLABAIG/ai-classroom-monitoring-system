"""
Behavioral & Discipline Risk Prediction (Advanced AI Feature 9.1)
--------------------------------------------------------------------
This is a transparent, rule-based EARLY-WARNING system, not a crystal
ball. Nothing here "predicts a misbehaviour" as a fact before it
happens -- that isn't something a classroom camera/database can
honestly promise. What it CAN do is turn patterns that are already
sitting in the database into an explainable per-student risk score,
so a teacher/admin can step in before a pattern escalates:

  - Attendance irregularity (absences, lateness) over a trailing window
  - Discipline flags a teacher has logged (disruptive behaviour,
    disrespect, conflict, repeated rule-breaking, etc.)
  - Whether flags are trending UP recently vs. before (escalation)

Every score comes with the plain-language "factors" that produced it,
so nobody has to trust a black-box number. Treat "High" as "worth a
conversation with the student," never as an automatic verdict or a
punishment trigger -- same philosophy as malpractice_ai.py's heuristic
signals.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------
# Tunable thresholds -- adjust here if scores feel too harsh/lenient
# ---------------------------------------------------------------------
LOOKBACK_DAYS = 30          # how far back we look for patterns
TREND_WINDOW_DAYS = 7       # "recent" vs "before that" comparison window
MAX_FLAG_SCORE = 55         # cap on points contributed by discipline flags
MAX_ATTENDANCE_SCORE = 25   # cap on points contributed by attendance
TREND_BONUS = 10            # points added when flags are escalating

FLAG_SEVERITY_WEIGHT = {"low": 4, "medium": 8, "high": 14}

FLAG_TYPES = [
    "Disruptive in class",
    "Talking back / disrespect",
    "Conflict or bullying",
    "Repeated rule-breaking",
    "Left classroom without permission",
    "Property damage",
    "Sleeping in class",
    "Phone use in class",
    "Other",
]

# Flag types a teacher can create with one click straight from an Activeness
# Monitor alert (see app.py's activeness_log_flag route). Kept separate from
# FLAG_TYPES-the-dropdown-list so the mapping is explicit and doesn't drift
# if the dropdown copy above ever changes wording.
AUTO_FLAG_TYPE_FOR_BAND = {
    "sleeping": "Sleeping in class",
    "phone_use_sustained": "Phone use in class",
}

HIGH_THRESHOLD = 55
MEDIUM_THRESHOLD = 28


def _level_for_score(score):
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def compute_student_risk(conn, student_id, today=None):
    """Compute one student's behavioral risk score + explanation.

    Returns a dict: score (0-100), level (Low/Medium/High), factors
    (list of plain-language strings), trend, flag_count, absence_rate,
    late_rate -- all derived only from data already in the database
    (attendance + discipline_flags), never invented.
    """
    today = today or date.today()
    lookback_start = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    trend_cutoff = (today - timedelta(days=TREND_WINDOW_DAYS)).isoformat()

    att_rows = conn.execute(
        "SELECT status FROM attendance WHERE student_id = ? AND att_date >= ?",
        (student_id, lookback_start),
    ).fetchall()
    total_days = len(att_rows)
    absent = sum(1 for r in att_rows if r["status"] == "Absent")
    late = sum(1 for r in att_rows if r["status"] == "Late")
    absence_rate = (absent / total_days) if total_days else 0.0
    late_rate = (late / total_days) if total_days else 0.0

    flag_rows = conn.execute(
        "SELECT severity, created_at FROM discipline_flags "
        "WHERE student_id = ? AND created_at >= ? ORDER BY created_at",
        (student_id, lookback_start),
    ).fetchall()

    sev_counts = {"low": 0, "medium": 0, "high": 0}
    flag_points = 0
    recent_count = 0
    prior_count = 0
    for r in flag_rows:
        sev = r["severity"] if r["severity"] in FLAG_SEVERITY_WEIGHT else "low"
        sev_counts[sev] += 1
        flag_points += FLAG_SEVERITY_WEIGHT[sev]
        if r["created_at"][:10] >= trend_cutoff:
            recent_count += 1
        else:
            prior_count += 1

    flag_score = min(flag_points, MAX_FLAG_SCORE)
    attendance_score = min(round(absence_rate * 20 + late_rate * 10), MAX_ATTENDANCE_SCORE)

    trend = "steady"
    trend_bonus = 0
    if not flag_rows:
        trend = None
    elif recent_count > prior_count and recent_count >= 2:
        trend = "rising"
        trend_bonus = TREND_BONUS
    elif recent_count < prior_count:
        trend = "easing"

    score = min(flag_score + attendance_score + trend_bonus, 100)
    level = _level_for_score(score)

    factors = []
    if flag_rows:
        parts = ", ".join(f"{c} {s}" for s, c in sev_counts.items() if c)
        factors.append(
            f"{len(flag_rows)} discipline flag(s) in the last {LOOKBACK_DAYS} days ({parts})."
        )
    if total_days and absent:
        factors.append(f"Absent {absent} of {total_days} recorded days ({round(absence_rate * 100)}%).")
    if total_days and late:
        factors.append(f"Late {late} of {total_days} recorded days ({round(late_rate * 100)}%).")
    if trend == "rising":
        factors.append(
            f"Flags are trending up: {recent_count} in the last {TREND_WINDOW_DAYS} days "
            f"vs {prior_count} in the {TREND_WINDOW_DAYS} days before that."
        )
    elif trend == "easing":
        factors.append(
            f"Flags are trending down: {recent_count} in the last {TREND_WINDOW_DAYS} days "
            f"vs {prior_count} before that."
        )
    if not factors:
        factors.append("No discipline flags or attendance irregularities on record.")

    return {
        "score": score,
        "level": level,
        "factors": factors,
        "trend": trend,
        "flag_count": len(flag_rows),
        "sev_counts": sev_counts,
        "absence_rate": round(absence_rate * 100),
        "late_rate": round(late_rate * 100),
        "total_days": total_days,
    }


def compute_class_risk(conn, class_id, today=None):
    """Risk scores for every student in a class, highest risk first."""
    students = conn.execute(
        "SELECT id, roll_no, name FROM students WHERE class_id = ? ORDER BY roll_no",
        (class_id,),
    ).fetchall()

    results = []
    for s in students:
        risk = compute_student_risk(conn, s["id"], today=today)
        risk["student_id"] = s["id"]
        risk["roll_no"] = s["roll_no"]
        risk["name"] = s["name"]
        results.append(risk)

    results.sort(key=lambda r: (-r["score"], r["roll_no"]))
    return results
