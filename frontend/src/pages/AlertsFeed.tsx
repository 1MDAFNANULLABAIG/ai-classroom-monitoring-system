import React, { useEffect, useState } from 'react';
import { AlertTriangle, ShieldCheck, XCircle, Eye, Video, CheckCircle, X } from 'lucide-react';
import { AlertItem } from '../types';
import { api } from '../services/api';

export const AlertsFeed: React.FC = () => {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [selectedSeverity, setSelectedSeverity] = useState<string>('');
  const [selectedType, setSelectedType] = useState<string>('');
  const [viewingAlert, setViewingAlert] = useState<AlertItem | null>(null);
  const [loading, setLoading] = useState(false);

  const loadAlerts = () => {
    setLoading(true);
    api
      .getAlerts(selectedSeverity || undefined, selectedType || undefined)
      .then((res) => setAlerts(res.alerts))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAlerts();
  }, [selectedSeverity, selectedType]);

  const handleAction = async (alertId: number, action: 'CONFIRMED' | 'DISMISSED') => {
    try {
      await api.takeAlertAction(alertId, action);
      setViewingAlert(null);
      loadAlerts();
      alert(`Alert marked as ${action}.`);
    } catch (err: any) {
      alert(`Error updating alert: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <span>Behavioral & Malpractice Evidence Vault</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real multi-photo and 10-second rolling video clips captured automatically for inattentiveness, phone usage, and examination malpractice.
          </p>
        </div>
      </div>

      {/* Filter Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label className="text-xs font-medium text-slate-400 block mb-1">Filter by Severity</label>
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-amber-500"
          >
            <option value="">All Severities</option>
            <option value="high">High Severity (Phone use, exam misconduct)</option>
            <option value="medium">Medium Severity (Repeated lookaway, head down)</option>
            <option value="low">Low Severity</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 block mb-1">Filter by Behavior</label>
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-amber-500"
          >
            <option value="">All Behaviors</option>
            <option value="PHONE_USAGE">Phone Usage</option>
            <option value="POSSIBLE_SLEEPING">Head Down / Sleeping</option>
            <option value="LOOKING_AWAY">Looking Away</option>
            <option value="LOOKING_BACK">Turning Back</option>
            <option value="LEAVING_SEAT">Leaving Seat</option>
            <option value="POSSIBLE_TALKING">Talking / Mouth Movement</option>
          </select>
        </div>
      </div>

      {/* Alerts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {alerts.length === 0 ? (
          <div className="col-span-full py-16 text-center text-slate-500 bg-slate-900 border border-slate-800 rounded-xl">
            <CheckCircle className="w-10 h-10 mx-auto mb-2 text-emerald-500/50" />
            <p className="text-sm">No malpractice or disengagement alerts matching criteria.</p>
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className="bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition space-y-3 relative overflow-hidden"
            >
              <div className="flex items-center justify-between">
                <span
                  className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${
                    alert.severity === 'high'
                      ? 'bg-red-950/60 text-red-300 border-red-800/40'
                      : alert.severity === 'medium'
                      ? 'bg-amber-950/60 text-amber-300 border-amber-800/40'
                      : 'bg-blue-950/60 text-blue-300 border-blue-800/40'
                  }`}
                >
                  {alert.severity} Severity
                </span>
                <span className="text-xs text-slate-400 font-mono">{alert.created_at}</span>
              </div>

              <div>
                <h3 className="text-base font-bold text-white">
                  {alert.student_name || 'Unidentified Student'}{' '}
                  {alert.usn && <span className="text-xs text-slate-400 font-normal">({alert.usn})</span>}
                </h3>
                <div className="text-xs font-semibold text-amber-400 font-mono mt-0.5">{alert.alert_type}</div>
                <p className="text-xs text-slate-300 mt-1">{alert.detail}</p>
              </div>

              {/* Status and Action */}
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between">
                <span className="text-xs text-slate-400">
                  Status:{' '}
                  <span
                    className={`font-semibold ${
                      alert.review_status === 'CONFIRMED'
                        ? 'text-red-400'
                        : alert.review_status === 'DISMISSED'
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                    }`}
                  >
                    {alert.review_status || 'REVIEW_REQUIRED'}
                  </span>
                </span>

                <button
                  onClick={() => setViewingAlert(alert)}
                  className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>Inspect Evidence</span>
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Evidence Viewer Modal */}
      {viewingAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl">
            <div className="p-4 sm:p-5 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center space-x-2">
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                  <span>Evidence Package: {viewingAlert.alert_type}</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Student: {viewingAlert.student_name} {viewingAlert.usn && `(${viewingAlert.usn})`} • Logged at: {viewingAlert.created_at}
                </p>
              </div>
              <button
                onClick={() => setViewingAlert(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-4 max-h-[80vh] overflow-y-auto">
              {/* Video Player if video evidence exists */}
              {viewingAlert.video_url && (
                <div>
                  <span className="text-xs font-semibold text-slate-300 block mb-1.5 flex items-center space-x-1.5">
                    <Video className="w-4 h-4 text-emerald-400" />
                    <span>10-Second Rolling Video Evidence Clip</span>
                  </span>
                  <div className="aspect-video bg-black rounded-xl overflow-hidden border border-slate-800">
                    <video controls autoPlay className="w-full h-full object-contain" src={viewingAlert.video_url} />
                  </div>
                </div>
              )}

              {/* Photo Evidence if available */}
              {viewingAlert.photo_url && (
                <div>
                  <span className="text-xs font-semibold text-slate-300 block mb-1.5">
                    Event Snapshot
                  </span>
                  <div className="aspect-[4/3] bg-black rounded-xl overflow-hidden border border-slate-800">
                    <img
                      src={viewingAlert.photo_url}
                      alt="Alert Evidence"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              )}

              {/* Details card */}
              <div className="bg-slate-850 p-3.5 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-1">
                <div>
                  <span className="font-semibold text-slate-400">Behavioral Description:</span> {viewingAlert.detail}
                </div>
                <div>
                  <span className="font-semibold text-slate-400">Severity Tier:</span> {viewingAlert.severity.toUpperCase()}
                </div>
              </div>

              {/* Decision Actions */}
              <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
                <span className="text-xs text-slate-400">Verify observable integrity:</span>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleAction(viewingAlert.id, 'DISMISSED')}
                    className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition"
                  >
                    <XCircle className="w-4 h-4 text-slate-400" />
                    <span>Dismiss False Positive</span>
                  </button>
                  <button
                    onClick={() => handleAction(viewingAlert.id, 'CONFIRMED')}
                    className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-bold transition shadow-md shadow-red-900/40"
                  >
                    <ShieldCheck className="w-4 h-4" />
                    <span>Confirm Malpractice</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
