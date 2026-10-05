import React, { useEffect, useState } from 'react';
import { 
  Users, CheckCircle2, AlertTriangle, GraduationCap, 
  Video, ArrowRight, ShieldAlert, Sparkles, TrendingUp 
} from 'lucide-react';
import { api } from '../services/api';
import { AnalyticsMetrics, AlertItem, ClassItem } from '../types';

interface DashboardProps {
  onNavigate: (tab: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getAnalytics().then(res => setMetrics(res.metrics)).catch(() => {}),
      api.getAlerts().then(res => setAlerts(res.alerts.slice(0, 5))).catch(() => {}),
      api.getClasses().then(res => setClasses(res.classes)).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-850 to-emerald-950 p-6 sm:p-8 border border-slate-800 shadow-xl">
        <div className="relative z-10 max-w-2xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI Multi-Modal Fusion Engine Active</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Classroom Intelligence & Malpractice Detection
          </h1>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">
            Real-time multi-person tracking, YuNet/SFace recognition, YOLO object & phone detection, and MediaPipe behavior estimation with instant duplicate-safe attendance recording.
          </p>
          <div className="mt-5 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('classroom')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold text-sm transition shadow-lg shadow-emerald-500/25"
            >
              <Video className="w-4 h-4 text-slate-950" />
              <span>Launch Live Camera</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button
              onClick={() => onNavigate('students')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm border border-slate-700 transition"
            >
              <Users className="w-4 h-4 text-emerald-400" />
              <span>Enroll Students</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Students */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Students</span>
            <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
              <Users className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white">{metrics?.total_students ?? 0}</div>
            <div className="text-xs text-slate-400 mt-1 flex items-center space-x-1">
              <span>Registered in system database</span>
            </div>
          </div>
        </div>

        {/* Today's Attendance */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Today's Attendance</span>
            <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-400">
              <CheckCircle2 className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-emerald-400">{metrics?.attendance_rate ?? 0}%</div>
            <div className="text-xs text-slate-400 mt-1 flex items-center space-x-1">
              <span>{metrics?.present_today ?? 0} Present</span>
              <span className="text-slate-600">•</span>
              <span>{metrics?.absent_today ?? 0} Absent</span>
            </div>
          </div>
        </div>

        {/* Teachers */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Faculty Teachers</span>
            <div className="p-2 bg-purple-500/10 rounded-lg text-purple-400">
              <GraduationCap className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white">{metrics?.total_teachers ?? 0}</div>
            <div className="text-xs text-slate-400 mt-1">
              <span>Teacher verification enabled</span>
            </div>
          </div>
        </div>

        {/* Malpractice Alerts */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Recent Alerts</span>
            <div className="p-2 bg-amber-500/10 rounded-lg text-amber-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-amber-400">{alerts.length}</div>
            <div className="text-xs text-slate-400 mt-1">
              <span>Evidence packages captured</span>
            </div>
          </div>
        </div>
      </div>

      {/* Two Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Quick Launch & Active Classes */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h2 className="text-base font-semibold text-white mb-3 flex items-center justify-between">
              <span>Available Classes</span>
              <span className="text-xs font-normal text-slate-400">{classes.length} Classes</span>
            </h2>
            <div className="space-y-2">
              {classes.slice(0, 5).map((c) => (
                <div
                  key={c.id}
                  onClick={() => onNavigate('classroom')}
                  className="p-3 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/50 cursor-pointer transition flex items-center justify-between group"
                >
                  <div>
                    <div className="text-sm font-medium text-slate-200 group-hover:text-emerald-400 transition">{c.name}</div>
                    <div className="text-xs text-slate-400">{c.department} • Sem {c.semester}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono">
                      {c.student_count || 0} students
                    </span>
                    <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 transition" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gradient-to-br from-slate-900 to-slate-850 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white flex items-center space-x-2 mb-2">
              <ShieldAlert className="w-4 h-4 text-emerald-400" />
              <span>Multi-Signal Safety Policy</span>
            </h3>
            <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
              <li>Attendance locked until teacher presence confirmed</li>
              <li>Duplicate attendance prevented for same student in period</li>
              <li>Phone usage detected via YOLO Ultralytics</li>
              <li>Sustained lookaway & head-down recorded with evidence</li>
            </ul>
          </div>
        </div>

        {/* Right: Recent Malpractice & Inattention Alerts */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold text-white">Latest Behavior & Malpractice Alerts</h2>
              <p className="text-xs text-slate-400">Evidence generated with video replay and candidate association</p>
            </div>
            <button
              onClick={() => onNavigate('alerts')}
              className="text-xs font-medium text-emerald-400 hover:text-emerald-300 transition flex items-center space-x-1"
            >
              <span>View all alerts</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {alerts.length === 0 ? (
            <div className="py-12 text-center text-slate-500">
              <CheckCircle2 className="w-10 h-10 mx-auto mb-2 text-emerald-500/50" />
              <p className="text-sm">No unresolved alerts recorded.</p>
              <p className="text-xs text-slate-600 mt-1">Classroom behavior running within normal parameters.</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {alerts.map((alert) => (
                <div key={alert.id} className="py-3 flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      alert.severity === 'high' ? 'bg-red-500/10 text-red-400' :
                      alert.severity === 'medium' ? 'bg-amber-500/10 text-amber-400' : 'bg-blue-500/10 text-blue-400'
                    }`}>
                      <AlertTriangle className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-200">
                        {alert.student_name || 'Unidentified Student'} {alert.usn && <span className="text-slate-400 text-xs">({alert.usn})</span>}
                      </div>
                      <div className="text-xs text-slate-400">{alert.alert_type} • {alert.detail || 'Observable behavior logged'}</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`text-[11px] px-2 py-0.5 rounded font-medium uppercase ${
                      alert.severity === 'high' ? 'bg-red-950/60 text-red-300 border border-red-800/40' :
                      alert.severity === 'medium' ? 'bg-amber-950/60 text-amber-300 border border-amber-800/40' : 'bg-slate-800 text-slate-300'
                    }`}>
                      {alert.severity}
                    </span>
                    <button
                      onClick={() => onNavigate('alerts')}
                      className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800"
                    >
                      Evidence
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
