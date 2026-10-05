import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingUp, CheckCircle2, AlertTriangle, Users, Calendar } from 'lucide-react';
import { AnalyticsMetrics } from '../types';
import { api } from '../services/api';

export const AnalyticsPage: React.FC = () => {
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getAnalytics()
      .then((res) => setMetrics(res.metrics))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h1 className="text-xl font-bold text-white flex items-center space-x-2">
          <BarChart3 className="w-5 h-5 text-emerald-400" />
          <span>Classroom & Examination Intelligence Analytics</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Historical attendance distributions, student engagement metrics, and alert category breakdowns calculated from SQLite records.
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Today's Attendance Rate</span>
          <div className="text-3xl font-extrabold text-emerald-400 mt-2">
            {metrics?.attendance_rate ?? 0}%
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {metrics?.present_today ?? 0} Present / {metrics?.absent_today ?? 0} Absent
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Total Registered Students</span>
          <div className="text-3xl font-extrabold text-white mt-2">
            {metrics?.total_students ?? 0}
          </div>
          <div className="text-xs text-slate-400 mt-1">Across all active classrooms</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Faculty Teachers</span>
          <div className="text-3xl font-extrabold text-purple-400 mt-2">
            {metrics?.total_teachers ?? 0}
          </div>
          <div className="text-xs text-slate-400 mt-1">Face authentication active</div>
        </div>
      </div>

      {/* Two Columns: Attendance Trend & Behavior Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Past 7 Days Attendance Trend */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <span>Weekly Attendance Trend</span>
            </h2>
            <span className="text-xs text-slate-400">Past 7 Days</span>
          </div>

          <div className="space-y-3 pt-2">
            {metrics?.weekly_trend?.map((item) => (
              <div key={item.date} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-slate-300">{item.date}</span>
                  <span className="font-bold text-emerald-400">{item.attendance_rate}%</span>
                </div>
                <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
                    style={{ width: `${item.attendance_rate}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Behavior & Malpractice Breakdown */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Behavior & Alert Category Breakdown</span>
          </h2>

          <div className="space-y-2.5 pt-2">
            {!metrics?.alert_breakdown || metrics.alert_breakdown.length === 0 ? (
              <div className="py-8 text-center text-xs text-slate-500">
                No alerts recorded in database yet.
              </div>
            ) : (
              metrics.alert_breakdown.map((item) => (
                <div
                  key={item.alert_type}
                  className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/50 flex items-center justify-between text-xs"
                >
                  <span className="font-semibold text-slate-200 font-mono">{item.alert_type}</span>
                  <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-bold font-mono">
                    {item.count} events
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
