import React, { useEffect, useState } from 'react';
import { Calendar, Plus, Clock, Trash2, X, CheckCircle2, AlertCircle } from 'lucide-react';
import { TimetablePeriod, ClassItem, Teacher } from '../types';
import { api } from '../services/api';

export const Timetable: React.FC = () => {
  const [periods, setPeriods] = useState<TimetablePeriod[]>([]);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [currentVerification, setCurrentVerification] = useState<{
    current_time: string;
    is_active_period: boolean;
    active_period: TimetablePeriod | null;
  } | null>(null);

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newPeriod, setNewPeriod] = useState({
    class_id: 1,
    teacher_id: 1,
    period_number: 1,
    period_type: 'class' as const,
    subject: '',
    start_time: '09:00',
    end_time: '09:50',
  });

  const loadData = () => {
    Promise.all([
      api.getClasses().then((res) => {
        setClasses(res.classes);
        if (res.classes.length > 0 && !selectedClassId) {
          setSelectedClassId(res.classes[0].id);
          setNewPeriod((p) => ({ ...p, class_id: res.classes[0].id }));
        }
      }),
      api.getTeachers().then((res) => {
        setTeachers(res.teachers);
        if (res.teachers.length > 0) {
          setNewPeriod((p) => ({ ...p, teacher_id: res.teachers[0].id }));
        }
      }),
      api.getTimetable(selectedClassId || undefined).then((res) => {
        setPeriods(res.timetable);
      }),
    ]);

    if (selectedClassId) {
      api.getCurrentPeriod(selectedClassId).then((res) => {
        setCurrentVerification({
          current_time: res.current_time,
          is_active_period: res.is_active_period,
          active_period: res.active_period,
        });
      });
    }
  };

  useEffect(() => {
    loadData();
  }, [selectedClassId]);

  const handleCreatePeriod = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.createTimetablePeriod(newPeriod);
      setIsAddModalOpen(false);
      setNewPeriod((p) => ({ ...p, subject: '' }));
      loadData();
      alert('Period successfully scheduled in timetable.');
    } catch (err: any) {
      alert(`Error scheduling period: ${err.message}`);
    }
  };

  const handleDeletePeriod = async (id: number) => {
    if (!confirm('Remove this scheduled period from the timetable?')) return;
    try {
      await api.deleteTimetablePeriod(id);
      loadData();
    } catch (err: any) {
      alert(`Error deleting period: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <Calendar className="w-5 h-5 text-emerald-400" />
            <span>Class Timetable & Schedule Verification</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Defines scheduled class hours and subject assignments. The live monitoring system validates these rules before unlocking student attendance.
          </p>
        </div>

        <button
          onClick={() => setIsAddModalOpen(true)}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold transition shadow-md shadow-emerald-900/30"
        >
          <Plus className="w-4 h-4" />
          <span>Add Timetable Period</span>
        </button>
      </div>

      {/* Class Selector & Real-Time Clock Verification Pill */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="w-full sm:w-auto">
          <select
            value={selectedClassId || ''}
            onChange={(e) => setSelectedClassId(Number(e.target.value))}
            className="bg-slate-900 border border-slate-800 text-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
          >
            {classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} ({c.department} - Sem {c.semester})
              </option>
            ))}
          </select>
        </div>

        {currentVerification && (
          <div className="w-full sm:w-auto flex items-center space-x-3 bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl text-xs">
            <Clock className="w-4 h-4 text-slate-400" />
            <span className="text-slate-400 font-mono">System Clock: {currentVerification.current_time}</span>
            {currentVerification.is_active_period ? (
              <span className="inline-flex items-center text-emerald-400 font-semibold bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                Active: {currentVerification.active_period?.subject}
              </span>
            ) : (
              <span className="inline-flex items-center text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                No active scheduled period right now
              </span>
            )}
          </div>
        )}
      </div>

      {/* Timetable Period Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {periods.length === 0 ? (
          <div className="col-span-full py-12 text-center text-slate-500 bg-slate-900 border border-slate-800 rounded-xl">
            No periods scheduled for this class. Click "Add Timetable Period" above.
          </div>
        ) : (
          periods.map((p) => {
            const isCurrentlyActive =
              currentVerification?.is_active_period &&
              currentVerification.active_period?.id === p.id;

            return (
              <div
                key={p.id}
                className={`bg-slate-900 border rounded-xl p-4 transition shadow-sm relative overflow-hidden ${
                  isCurrentlyActive
                    ? 'border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.15)] bg-slate-900/90'
                    : 'border-slate-800 hover:border-slate-700'
                }`}
              >
                {isCurrentlyActive && (
                  <div className="absolute top-0 right-0 bg-emerald-500 text-slate-950 text-[10px] font-bold px-2.5 py-0.5 rounded-bl-lg tracking-wider">
                    CURRENTLY ACTIVE
                  </div>
                )}

                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    Period #{p.period_number} • {p.period_type.toUpperCase()}
                  </span>
                  <button
                    onClick={() => handleDeletePeriod(p.id)}
                    className="p-1 rounded text-slate-500 hover:text-red-400 hover:bg-slate-800 transition"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                <h3 className="text-base font-bold text-white mt-1">{p.subject}</h3>
                <div className="text-xs text-slate-400 mt-0.5">Faculty: {p.teacher_name || 'Unassigned'}</div>

                <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-mono text-slate-300">
                  <div className="flex items-center space-x-1.5">
                    <Clock className="w-3.5 h-3.5 text-emerald-400" />
                    <span>
                      {p.start_time} - {p.end_time}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Add Period Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Add Scheduled Period</h3>
              <button
                onClick={() => setIsAddModalOpen(false)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <form onSubmit={handleCreatePeriod} className="p-5 space-y-3.5">
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Class</label>
                <select
                  value={newPeriod.class_id}
                  onChange={(e) => setNewPeriod({ ...newPeriod, class_id: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                >
                  {classes.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Subject Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Computer Networks"
                  value={newPeriod.subject}
                  onChange={(e) => setNewPeriod({ ...newPeriod, subject: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Assigned Teacher</label>
                <select
                  value={newPeriod.teacher_id}
                  onChange={(e) => setNewPeriod({ ...newPeriod, teacher_id: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                >
                  {teachers.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name} ({t.teacher_code})
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-1">Start Time (HH:MM)</label>
                  <input
                    type="time"
                    required
                    value={newPeriod.start_time}
                    onChange={(e) => setNewPeriod({ ...newPeriod, start_time: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-1">End Time (HH:MM)</label>
                  <input
                    type="time"
                    required
                    value={newPeriod.end_time}
                    onChange={(e) => setNewPeriod({ ...newPeriod, end_time: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div className="pt-2 flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setIsAddModalOpen(false)}
                  className="px-4 py-2 rounded-lg bg-slate-800 text-slate-300 text-xs font-medium hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition"
                >
                  Save Schedule
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
