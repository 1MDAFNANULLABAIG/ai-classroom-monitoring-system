import React, { useEffect, useState } from 'react';
import { 
  LayoutDashboard, Video, Users, GraduationCap, Calendar, 
  FileCheck2, AlertTriangle, BarChart3, LogOut, Activity, Cpu 
} from 'lucide-react';
import { User } from '../types';
import { api } from '../services/api';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  user: User | null;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, setCurrentTab, user, onLogout }) => {
  const [aiStatus, setAiStatus] = useState<boolean | null>(null);

  useEffect(() => {
    api.getHealth()
      .then(res => setAiStatus(res.ai_engine?.yolo_ready && res.ai_engine?.sface_ready))
      .catch(() => setAiStatus(false));
  }, []);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'classroom', label: 'Live Classroom', icon: Video, highlight: true },
    { id: 'students', label: 'Students', icon: Users },
    { id: 'teachers', label: 'Teachers', icon: GraduationCap },
    { id: 'timetable', label: 'Timetable', icon: Calendar },
    { id: 'attendance', label: 'Attendance', icon: FileCheck2 },
    { id: 'alerts', label: 'Malpractice Alerts', icon: AlertTriangle },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setCurrentTab('dashboard')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-900/40">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-lg font-bold bg-gradient-to-r from-white via-slate-200 to-emerald-400 bg-clip-text text-transparent">
                Classroom AI
              </span>
              <span className="text-xs block text-slate-400 font-mono">Real-Time Intelligence</span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? item.highlight 
                        ? 'bg-emerald-600 text-white shadow-md shadow-emerald-900/50'
                        : 'bg-slate-800 text-emerald-400'
                      : item.highlight
                        ? 'text-emerald-400 hover:bg-emerald-950/40'
                        : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive && item.highlight ? 'text-white' : ''}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Status & User */}
          <div className="flex items-center space-x-4">
            {/* AI Engine Status Pill */}
            <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-xs text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              <span>YOLO & SFace:</span>
              {aiStatus === true ? (
                <span className="inline-flex items-center text-emerald-400 font-semibold">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse mr-1" /> Ready
                </span>
              ) : aiStatus === false ? (
                <span className="text-red-400 font-medium">Degraded</span>
              ) : (
                <span className="text-slate-400">Loading...</span>
              )}
            </div>

            {/* User Pill */}
            {user && (
              <div className="flex items-center space-x-3 pl-2 border-l border-slate-800">
                <div className="text-right hidden sm:block">
                  <div className="text-xs font-semibold text-slate-200">{user.username}</div>
                  <div className="text-[10px] text-emerald-400 uppercase tracking-wider font-mono">{user.role}</div>
                </div>
                <button
                  onClick={onLogout}
                  title="Sign out"
                  className="p-2 rounded-lg bg-slate-800 hover:bg-red-950/40 text-slate-400 hover:text-red-400 transition"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
