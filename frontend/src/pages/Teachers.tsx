import React, { useEffect, useState } from 'react';
import { GraduationCap, Plus, Camera, CheckCircle2, Clock, X } from 'lucide-react';
import { Teacher } from '../types';
import { api } from '../services/api';
import { TeacherEnrollmentModal } from './TeacherEnrollmentModal';

export const Teachers: React.FC = () => {
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [enrollingTeacher, setEnrollingTeacher] = useState<Teacher | null>(null);

  const [newTeacher, setNewTeacher] = useState({
    name: '',
    teacher_code: '',
    email: '',
    phone: '',
  });

  const loadTeachers = () => {
    api.getTeachers().then((res) => {
      setTeachers(res.teachers);
    });
  };

  useEffect(() => {
    loadTeachers();
  }, []);

  const handleCreateTeacher = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.createTeacher(newTeacher);
      setIsAddModalOpen(false);
      setNewTeacher({ name: '', teacher_code: '', email: '', phone: '' });
      loadTeachers();
      alert('Teacher registered successfully! You can now enroll their face.');
    } catch (err: any) {
      alert(`Error registering teacher: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <GraduationCap className="w-5 h-5 text-purple-400" />
            <span>Faculty & Teacher Verification</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Teacher-first attendance security: Attendance is unlocked only when the assigned teacher is face-confirmed present.
          </p>
        </div>

        <button
          onClick={() => setIsAddModalOpen(true)}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-sm font-semibold transition shadow-md shadow-purple-900/30"
        >
          <Plus className="w-4 h-4" />
          <span>Add New Teacher</span>
        </button>
      </div>

      {/* Teachers Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-slate-950 text-xs uppercase font-semibold text-slate-400 tracking-wider border-b border-slate-800">
            <tr>
              <th className="px-5 py-3.5">Teacher Code</th>
              <th className="px-5 py-3.5">Teacher Name</th>
              <th className="px-5 py-3.5">Contact</th>
              <th className="px-5 py-3.5">Face Samples</th>
              <th className="px-5 py-3.5">Status</th>
              <th className="px-5 py-3.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {teachers.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-12 text-slate-500">
                  No teachers registered yet. Click "Add New Teacher".
                </td>
              </tr>
            ) : (
              teachers.map((t) => (
                <tr key={t.id} className="hover:bg-slate-850/50 transition">
                  <td className="px-5 py-4 font-mono font-medium text-purple-400">
                    {t.teacher_code}
                  </td>
                  <td className="px-5 py-4">
                    <div className="font-semibold text-white">{t.name}</div>
                  </td>
                  <td className="px-5 py-4 text-xs text-slate-400">
                    <div>{t.email || 'No email registered'}</div>
                    <div>{t.phone || 'No phone registered'}</div>
                  </td>
                  <td className="px-5 py-4">
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs font-mono font-semibold ${
                        t.sample_count >= 50 ? 'text-purple-400' : 'text-amber-400'
                      }`}>
                        {t.sample_count} / 50
                      </span>
                      <div className="w-16 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${t.sample_count >= 50 ? 'bg-purple-500' : 'bg-amber-500'}`}
                          style={{ width: `${Math.min(100, (t.sample_count / 50) * 100)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    {t.face_enrolled ? (
                      <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Enrolled</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                        <Clock className="w-3.5 h-3.5" />
                        <span>Pending</span>
                      </span>
                    )}
                  </td>
                  <td className="px-5 py-4 text-right">
                    <button
                      onClick={() => setEnrollingTeacher(t)}
                      className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-purple-600 hover:text-white text-slate-200 text-xs font-semibold border border-slate-700 hover:border-purple-500 transition"
                    >
                      <Camera className="w-3.5 h-3.5" />
                      <span>Enroll Face</span>
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Add Teacher Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Register Faculty Teacher</h3>
              <button
                onClick={() => setIsAddModalOpen(false)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <form onSubmit={handleCreateTeacher} className="p-5 space-y-3.5">
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Teacher Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Dr. Jane Smith"
                  value={newTeacher.name}
                  onChange={(e) => setNewTeacher({ ...newTeacher, name: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Teacher Code / Employee ID</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. TCH001"
                  value={newTeacher.teacher_code}
                  onChange={(e) => setNewTeacher({ ...newTeacher, teacher_code: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Email</label>
                <input
                  type="email"
                  placeholder="teacher@institution.edu"
                  value={newTeacher.email}
                  onChange={(e) => setNewTeacher({ ...newTeacher, email: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Phone</label>
                <input
                  type="text"
                  placeholder="+91 9876543210"
                  value={newTeacher.phone}
                  onChange={(e) => setNewTeacher({ ...newTeacher, phone: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-purple-500"
                />
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
                  className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-xs font-bold transition"
                >
                  Save & Register
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {enrollingTeacher && (
        <TeacherEnrollmentModal
          teacher={enrollingTeacher}
          onClose={() => setEnrollingTeacher(null)}
          onEnrollmentComplete={() => {
            loadTeachers();
          }}
        />
      )}
    </div>
  );
};
