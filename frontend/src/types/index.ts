export interface User {
  id: number;
  username: string;
  role: 'admin' | 'teacher';
}

export interface ClassItem {
  id: number;
  name: string;
  start_time: string;
  end_time: string;
  department: string;
  semester: number;
  student_count?: number;
}

export interface Student {
  id: number;
  name: string;
  roll_no: string;
  usn: string;
  class_id: number;
  class_name?: string;
  email: string;
  phone: string;
  department?: string;
  semester?: number;
  section?: string;
  admission_year?: string;
  status: string;
  sample_count: number;
  face_enrolled: boolean;
}

export interface Teacher {
  id: number;
  name: string;
  teacher_code: string;
  email: string;
  phone: string;
  sample_count: number;
  face_enrolled: boolean;
}

export interface TimetablePeriod {
  id: number;
  class_id: number;
  class_name?: string;
  teacher_id?: number;
  teacher_name?: string;
  period_number: number;
  period_type: 'class' | 'break' | 'lab';
  subject: string;
  start_time: string;
  end_time: string;
}

export interface ClassroomSession {
  session_id: number;
  class_id: number;
  subject: string;
  started_at: string;
  teacher_confirmed: boolean;
  teacher_name: string;
  total_students: number;
  present_count: number;
  absent_count: number;
  attendance_percentage: number;
  alerts_count: number;
}

export interface AlertItem {
  id: number;
  session_id: number;
  student_id?: number;
  student_name?: string;
  usn?: string;
  alert_type: string;
  severity: 'low' | 'medium' | 'high';
  detail: string;
  photo_path?: string;
  photo_url?: string;
  video_path?: string;
  video_url?: string;
  review_status?: string;
  created_at: string;
  source?: string;
}

export interface TrackedPerson {
  track_id: number;
  box: [number, number, number, number]; // [x, y, w, h]
  name: string;
  usn: string;
  status: 'FACE_VERIFIED' | 'TRACK_ASSOCIATED' | 'UNKNOWN';
  identity_source: string;
  confidence: number;
  activity: string;
  attendance_marked: boolean;
}

export interface DetectedObject {
  label: string;
  box: [number, number, number, number];
  conf: number;
}

export interface DetectionPayload {
  type: string;
  timestamp: string;
  fps: {
    e2e_fps: number;
    detection_fps: number;
    recognition_fps: number;
    latency_ms: number;
  };
  counts: {
    total: number;
    present: number;
    absent: number;
    percentage: number;
  };
  tracked: TrackedPerson[];
  objects: DetectedObject[];
  teacher_confirmed: boolean;
  teacher_name: string;
}

export interface AttendanceRecord {
  id: number;
  att_date: string;
  status: 'present' | 'absent';
  student_id: number;
  student_name: string;
  roll_no: string;
  usn: string;
  class_name: string;
}

export interface AnalyticsMetrics {
  total_students: number;
  total_teachers: number;
  total_classes: number;
  present_today: number;
  absent_today: number;
  attendance_rate: number;
  alert_breakdown: { alert_type: string; count: number }[];
  weekly_trend: { date: string; attendance_rate: number; present: number; total: number }[];
}
