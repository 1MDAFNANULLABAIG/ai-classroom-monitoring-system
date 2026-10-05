import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { LiveClassroom } from './pages/LiveClassroom';
import { Students } from './pages/Students';
import { Teachers } from './pages/Teachers';
import { Timetable } from './pages/Timetable';
import { AttendanceRecords } from './pages/AttendanceRecords';
import { AlertsFeed } from './pages/AlertsFeed';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { Settings } from './pages/Settings';
import { AboutProject } from './pages/AboutProject';
import { Login } from './pages/Login';
import { User } from './types';
import { api, getAuthToken } from './services/api';

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [currentTab, setCurrentTab] = useState<string>('dashboard');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const token = getAuthToken();
    if (token) {
      api
        .getCurrentUser()
        .then((res) => {
          if (res.success) {
            setUser(res.user);
          }
        })
        .catch(() => {
          api.logout();
          setUser(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const handleLogout = () => {
    api.logout();
    setUser(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-400">
        <div className="flex items-center space-x-2 text-sm font-mono">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping mr-2" />
          Loading Smart Classroom Environment...
        </div>
      </div>
    );
  }

  if (!user) {
    return <Login onLoginSuccess={(u) => setUser(u)} />;
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        user={user}
        onLogout={handleLogout}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentTab === 'dashboard' && <Dashboard onNavigate={(tab) => setCurrentTab(tab)} />}
        {currentTab === 'classroom' && <LiveClassroom />}
        {currentTab === 'students' && <Students />}
        {currentTab === 'teachers' && <Teachers />}
        {currentTab === 'timetable' && <Timetable />}
        {currentTab === 'attendance' && <AttendanceRecords />}
        {currentTab === 'alerts' && <AlertsFeed />}
        {currentTab === 'analytics' && <AnalyticsPage />}
        {currentTab === 'settings' && <Settings />}
        {currentTab === 'about' && <AboutProject />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500">
        AI-Powered Smart Classroom Monitoring, Attendance & Malpractice Detection System • Production Build
      </footer>
    </div>
  );
}

export default App;
