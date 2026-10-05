import { 
  User, ClassItem, Student, Teacher, TimetablePeriod, 
  ClassroomSession, AlertItem, AttendanceRecord, AnalyticsMetrics 
} from '../types';

const API_BASE = '/api';

export const getAuthToken = (): string | null => {
  return localStorage.getItem('token');
};

export const setAuthToken = (token: string | null) => {
  if (token) {
    localStorage.setItem('token', token);
  } else {
    localStorage.removeItem('token');
  }
};

async function apiFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || data.message || `Request failed (${res.status})`);
  }
  return data as T;
}

export const api = {
  // Health
  getHealth: () => apiFetch<{ status: string; ai_engine: any }>('/health'),

  // Auth
  login: async (username: string, password: string): Promise<{ success: boolean; token: string; user: User }> => {
    const res = await apiFetch<{ success: boolean; token: string; user: User }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    setAuthToken(res.token);
    return res;
  },
  getCurrentUser: (): Promise<{ success: boolean; user: User }> => apiFetch('/auth/me'),
  logout: () => setAuthToken(null),

  // Classes
  getClasses: (): Promise<{ success: boolean; classes: ClassItem[] }> => apiFetch('/classes'),
  createClass: (data: Partial<ClassItem>) => apiFetch('/classes', { method: 'POST', body: JSON.stringify(data) }),

  // Students
  getStudents: (classId?: number, search?: string): Promise<{ success: boolean; students: Student[] }> => {
    const params = new URLSearchParams();
    if (classId) params.append('class_id', classId.toString());
    if (search) params.append('search', search);
    return apiFetch(`/students?${params.toString()}`);
  },
  createStudent: (data: Partial<Student>) => apiFetch('/students', { method: 'POST', body: JSON.stringify(data) }),
  getStudentDetail: (id: number): Promise<{ success: boolean; student: any }> => apiFetch(`/students/${id}`),
  enrollStudentFrame: (studentId: number, imageBase64: string): Promise<{
    success: boolean;
    count: number;
    required: number;
    pose: string;
    quality_score: number;
    auto_trained: boolean;
    message: string;
  }> => apiFetch(`/students/${studentId}/enroll-frame`, {
    method: 'POST',
    body: JSON.stringify({ image: imageBase64 }),
  }),
  trainStudentModel: (studentId: number) => apiFetch(`/students/${studentId}/train`, { method: 'POST' }),

  // Teachers
  getTeachers: (): Promise<{ success: boolean; teachers: Teacher[] }> => apiFetch('/teachers'),
  createTeacher: (data: Partial<Teacher>) => apiFetch('/teachers', { method: 'POST', body: JSON.stringify(data) }),
  enrollTeacherFrame: (teacherId: number, imageBase64: string): Promise<{
    success: boolean;
    count: number;
    required: number;
    pose: string;
    quality_score: number;
    auto_trained: boolean;
    message: string;
  }> => apiFetch(`/teachers/${teacherId}/enroll-frame`, {
    method: 'POST',
    body: JSON.stringify({ image: imageBase64 }),
  }),
  trainTeacherModel: (teacherId: number) => apiFetch(`/teachers/${teacherId}/train`, { method: 'POST' }),

  // Timetable
  getTimetable: (classId?: number): Promise<{ success: boolean; timetable: TimetablePeriod[] }> => {
    const params = classId ? `?class_id=${classId}` : '';
    return apiFetch(`/timetable${params}`);
  },
  createTimetablePeriod: (data: Partial<TimetablePeriod>) => apiFetch('/timetable', { method: 'POST', body: JSON.stringify(data) }),
  deleteTimetablePeriod: (periodId: number) => apiFetch(`/timetable/${periodId}`, { method: 'DELETE' }),
  getCurrentPeriod: (classId: number): Promise<{
    success: boolean;
    current_time: string;
    is_active_period: boolean;
    active_period: TimetablePeriod | null;
    upcoming_period: TimetablePeriod | null;
  }> => apiFetch(`/timetable/current?class_id=${classId}`),

  // Live Classroom Session
  getClassroomStatus: (classId: number): Promise<{ success: boolean; is_active: boolean; session: ClassroomSession | null }> => 
    apiFetch(`/classroom/status?class_id=${classId}`),
  startClassroomSession: (classId: number, subject?: string, forceStart: boolean = false): Promise<{
    success: boolean;
    message: string;
    session: any;
  }> => apiFetch('/classroom/start', {
    method: 'POST',
    body: JSON.stringify({ class_id: classId, subject, force_start: forceStart }),
  }),
  stopClassroomSession: (classId: number): Promise<{ success: boolean; message: string; summary: any }> =>
    apiFetch('/classroom/stop', {
      method: 'POST',
      body: JSON.stringify({ class_id: classId }),
    }),

  // Attendance
  getAttendance: (params: { class_id?: number; date?: string; status?: string }): Promise<{ success: boolean; attendance: AttendanceRecord[] }> => {
    const q = new URLSearchParams();
    if (params.class_id) q.append('class_id', params.class_id.toString());
    if (params.date) q.append('date', params.date);
    if (params.status) q.append('status', params.status);
    return apiFetch(`/attendance?${q.toString()}`);
  },
  getExportUrl: (classId?: number, date?: string) => {
    const q = new URLSearchParams();
    if (classId) q.append('class_id', classId.toString());
    if (date) q.append('date', date);
    return `/api/attendance/export?${q.toString()}`;
  },

  // Alerts
  getAlerts: (severity?: string, type?: string): Promise<{ success: boolean; alerts: AlertItem[] }> => {
    const q = new URLSearchParams();
    if (severity) q.append('severity', severity);
    if (type) q.append('type', type);
    return apiFetch(`/alerts?${q.toString()}`);
  },
  takeAlertAction: (alertId: number, action: 'CONFIRMED' | 'DISMISSED', notes: string = '') =>
    apiFetch(`/alerts/${alertId}/action`, {
      method: 'POST',
      body: JSON.stringify({ action, notes }),
    }),

  // Analytics
  getAnalytics: (): Promise<{ success: boolean; metrics: AnalyticsMetrics }> => apiFetch('/analytics'),
};
