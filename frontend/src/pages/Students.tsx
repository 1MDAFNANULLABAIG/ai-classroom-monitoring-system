import React, { useEffect, useState } from 'react';
import { Users, Plus, Search, Camera, CheckCircle2, Clock, Trash2, X } from 'lucide-react';
import { Student, ClassItem } from '../types';
import { api } from '../services/api';
import { StudentEnrollmentModal } from './StudentEnrollmentModal';

export const Students: React.FC = () => {
  const [students, setStudents] = useState<Student[]>([]);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  // Modals state
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [enrollingStudent, setEnrollingStudent] = useState<Student | null>(null);

  // New Student Form
  const [newStudent, setNewStudent] = useState({
    name: '',
    roll_no: '',
    usn: '',
    class_id: 1,
    email: '',
    phone: '',
  });

  const loadData = () => {
    setLoading(true);
    Promise.all([
      api.getClasses().then((res) => {
        setClasses(res.classes);
        if (res.classes.length > 0 && !selectedClassId) {
          setNewStudent((prev) => ({ ...prev, class_id: res.classes[0].id }));
        }
      }),
      api.getStudents(selectedClassId || undefined, searchQuery).then((res) => {
        setStudents(res.students);
      }),
    ]).finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, [selectedClassId, searchQuery]);

  const handleCreateStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.createStudent({
        name: newStudent.name,
        roll_no: newStudent.roll_no,
        usn: newStudent.usn || newStudent.roll_no,
        class_id: newStudent.class_id,
        email: newStudent.email,
        phone: newStudent.phone,
      });
      setIsAddModalOpen(false);
      setNewStudent({ name: '', roll_no: '', usn: '', class_id: classes[0]?.id || 1, email: '', phone: '' });
      loadData();
      alert('Student registered successfully! Click "Enroll Face" to capture face samples.');
    } catch (err: any) {
      alert(`Error registering student: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <Users className="w-5 h-5 text-emerald-400" />
            <span>Student Management & Face Database</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Register students and record 50-sample multi-angle face profiles for instant real-time classroom recognition.
          </p>
        </div>

        <button
          onClick={() => setIsAddModalOpen(true)}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold transition shadow-md shadow-emerald-900/30"
        >
          <Plus className="w-4 h-4" />
          <span>Add New Student</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Search by student name, roll number, or USN..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="w-full sm:w-auto">
          <select
            value={selectedClassId || ''}
            onChange={(e) => setSelectedClassId(e.target.value ? Number(e.target.value) : null)}
            className="w-full sm:w-auto bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="">All Classes</option>
            {classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} ({c.department} - Sem {c.semester})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Students Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-950 text-xs uppercase font-semibold text-slate-400 tracking-wider border-b border-slate-800">
              <tr>
                <th className="px-5 py-3.5">Roll No & USN</th>
                <th className="px-5 py-3.5">Student Name</th>
                <th className="px-5 py-3.5">Class / Department</th>
                <th className="px-5 py-3.5">Face Samples</th>
                <th className="px-5 py-3.5">Status</th>
                <th className="px-5 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {students.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-slate-500">
                    No students registered yet. Click "Add New Student" above.
                  </td>
                </tr>
              ) : (
                students.map((st) => (
                  <tr key={st.id} className="hover:bg-slate-850/50 transition">
                    <td className="px-5 py-4 font-mono font-medium text-slate-200">
                      <div>{st.roll_no}</div>
                      <div className="text-[11px] text-slate-400">{st.usn}</div>
                    </td>
                    <td className="px-5 py-4">
                      <div className="font-semibold text-white">{st.name}</div>
                      <div className="text-xs text-slate-400">{st.email || 'No email registered'}</div>
                    </td>
                    <td className="px-5 py-4 text-xs text-slate-300">
                      {st.class_name || 'Class Assigned'}
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs font-mono font-semibold ${
                          st.sample_count >= 50 ? 'text-emerald-400' : 'text-amber-400'
                        }`}>
                          {st.sample_count} / 50
                        </span>
                        <div className="w-16 h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${st.sample_count >= 50 ? 'bg-emerald-500' : 'bg-amber-500'}`}
                            style={{ width: `${Math.min(100, (st.sample_count / 50) * 100)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-5 py-4">
                      {st.face_enrolled ? (
                        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Enrolled</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                          <Clock className="w-3.5 h-3.5" />
                          <span>Pending Face</span>
                        </span>
                      )}
                    </td>
                    <td className="px-5 py-4 text-right">
                      <button
                        onClick={() => setEnrollingStudent(st)}
                        className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 text-xs font-semibold border border-slate-700 hover:border-emerald-500 transition"
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
      </div>

      {/* Add Student Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Register New Student</h3>
              <button
                onClick={() => setIsAddModalOpen(false)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <form onSubmit={handleCreateStudent} className="p-5 space-y-3.5">
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Full Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. John Doe"
                  value={newStudent.name}
                  onChange={(e) => setNewStudent({ ...newStudent, name: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-1">Roll Number</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. 01"
                    value={newStudent.roll_no}
                    onChange={(e) => setNewStudent({ ...newStudent, roll_no: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-1">USN</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. 1HK23IS001"
                    value={newStudent.usn}
                    onChange={(e) => setNewStudent({ ...newStudent, usn: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Assigned Class</label>
                <select
                  value={newStudent.class_id}
                  onChange={(e) => setNewStudent({ ...newStudent, class_id: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                >
                  {classes.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ({c.department} - Sem {c.semester})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">Email (Optional)</label>
                <input
                  type="email"
                  placeholder="student@college.edu"
                  value={newStudent.email}
                  onChange={(e) => setNewStudent({ ...newStudent, email: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
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
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition"
                >
                  Save & Register
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Face Enrollment Modal */}
      {enrollingStudent && (
        <StudentEnrollmentModal
          student={enrollingStudent}
          onClose={() => setEnrollingStudent(null)}
          onEnrollmentComplete={() => {
            loadData();
          }}
        />
      )}
    </div>
  );
};
