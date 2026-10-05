import React, { useEffect, useState } from 'react';
import { FileCheck2, Download, Search, CheckCircle2, XCircle } from 'lucide-react';
import { AttendanceRecord, ClassItem } from '../types';
import { api } from '../services/api';

export const AttendanceRecords: React.FC = () => {
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [selectedStatus, setSelectedStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getClasses().then((res) => setClasses(res.classes));
  }, []);

  const loadAttendance = () => {
    setLoading(true);
    api
      .getAttendance({
        class_id: selectedClassId || undefined,
        date: selectedDate,
        status: selectedStatus || undefined,
      })
      .then((res) => setRecords(res.attendance))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAttendance();
  }, [selectedClassId, selectedDate, selectedStatus]);

  const handleExportCsv = () => {
    const url = api.getExportUrl(selectedClassId || undefined, selectedDate);
    window.open(url, '_blank');
  };

  const presentCount = records.filter((r) => r.status === 'present').length;
  const absentCount = records.filter((r) => r.status === 'absent').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <FileCheck2 className="w-5 h-5 text-emerald-400" />
            <span>Attendance Logs & Audit Trail</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Verified attendance records marked with multi-frame face recognition and SQLite transactions.
          </p>
        </div>

        <button
          onClick={handleExportCsv}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-sm font-semibold transition shadow-sm"
        >
          <Download className="w-4 h-4 text-emerald-400" />
          <span>Export CSV Report</span>
        </button>
      </div>

      {/* Filter Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label className="text-xs font-medium text-slate-400 block mb-1">Date</label>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 block mb-1">Class</label>
          <select
            value={selectedClassId || ''}
            onChange={(e) => setSelectedClassId(e.target.value ? Number(e.target.value) : null)}
            className="w-full bg-slate-900 border border-slate-800 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
          >
            <option value="">All Classes</option>
            {classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 block mb-1">Attendance Status</label>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
          >
            <option value="">All (Present & Absent)</option>
            <option value="present">Present Only</option>
            <option value="absent">Absent Only</option>
          </select>
        </div>
      </div>

      {/* Quick Stats Pill */}
      <div className="flex items-center space-x-4 text-xs font-mono">
        <span className="text-slate-400">Total Scored: {records.length}</span>
        <span className="text-emerald-400 font-bold">• {presentCount} Present</span>
        <span className="text-red-400 font-bold">• {absentCount} Absent</span>
      </div>

      {/* Records Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-950 text-xs uppercase font-semibold text-slate-400 tracking-wider border-b border-slate-800">
              <tr>
                <th className="px-5 py-3.5">Date</th>
                <th className="px-5 py-3.5">Roll No & USN</th>
                <th className="px-5 py-3.5">Student Name</th>
                <th className="px-5 py-3.5">Class</th>
                <th className="px-5 py-3.5">Attendance Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {records.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-12 text-slate-500">
                    No attendance records found matching filters.
                  </td>
                </tr>
              ) : (
                records.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-850/50 transition">
                    <td className="px-5 py-4 font-mono text-xs text-slate-400">{r.att_date}</td>
                    <td className="px-5 py-4 font-mono font-medium text-slate-200">
                      <div>{r.roll_no}</div>
                      <div className="text-[11px] text-slate-400">{r.usn}</div>
                    </td>
                    <td className="px-5 py-4 font-semibold text-white">{r.student_name}</td>
                    <td className="px-5 py-4 text-xs text-slate-300">{r.class_name}</td>
                    <td className="px-5 py-4">
                      {r.status === 'present' ? (
                        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>PRESENT</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-red-500/10 text-red-400 border border-red-500/20">
                          <XCircle className="w-3.5 h-3.5" />
                          <span>ABSENT</span>
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
