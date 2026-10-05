import React, { useState } from 'react';
import { Settings as SettingsIcon, Camera, Sliders, Shield, Bell, Database, RefreshCw, CheckCircle2, HardDrive } from 'lucide-react';

export const Settings: React.FC = () => {
  const [cameraSource, setCameraSource] = useState<string>('0');
  const [resolution, setResolution] = useState<string>('640x480');
  const [targetFps, setTargetFps] = useState<number>(30);
  const [faceThreshold, setFaceThreshold] = useState<number>(0.363);
  const [detectorConfidence, setDetectorConfidence] = useState<number>(0.70);
  const [yoloConfidence, setYoloConfidence] = useState<number>(0.25);
  const [lateThresholdMinutes, setLateThresholdMinutes] = useState<number>(10);
  const [autoMarkAbsent, setAutoMarkAbsent] = useState<boolean>(true);
  const [soundAlerts, setSoundAlerts] = useState<boolean>(true);
  const [savedSuccess, setSavedSuccess] = useState<boolean>(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem('classroom_ai_settings', JSON.stringify({
      cameraSource,
      resolution,
      targetFps,
      faceThreshold,
      detectorConfidence,
      yoloConfidence,
      lateThresholdMinutes,
      autoMarkAbsent,
      soundAlerts
    }));
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <SettingsIcon className="w-5 h-5 text-emerald-400" />
            <span>AI System & Classroom Monitoring Configuration</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Configure physical camera devices, AI recognition sensitivity, attendance policy, and alert thresholds.
          </p>
        </div>

        {savedSuccess && (
          <div className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/80 border border-emerald-800 text-emerald-400 text-xs font-semibold animate-pulse">
            <CheckCircle2 className="w-4 h-4" />
            <span>Settings saved successfully!</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Hardware & Camera Stream */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-white font-semibold text-sm border-b border-slate-800 pb-3">
            <Camera className="w-4 h-4 text-emerald-400" />
            <span>Camera & Stream Acquisition Settings</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Active Camera Input Device
              </label>
              <select
                value={cameraSource}
                onChange={(e) => setCameraSource(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="0">Default USB / Integrated Webcam (Index 0)</option>
                <option value="1">Secondary USB Classroom Cam (Index 1)</option>
                <option value="2">External Wide-Angle PTZ Cam (Index 2)</option>
                <option value="browser">Browser Client Stream (navigator.mediaDevices)</option>
              </select>
              <p className="text-[11px] text-slate-500 mt-1">
                Hardware webcam accessed directly by edge OpenCV loop.
              </p>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                AI Frame Capture Resolution
              </label>
              <select
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="640x480">640 x 480 (Recommended - High FPS & Low Latency)</option>
                <option value="1280x720">1280 x 720 (HD - Extended Depth Range)</option>
                <option value="1920x1080">1920 x 1080 (Full HD - Large Lecture Halls)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Target Capture Rate: {targetFps} FPS
              </label>
              <input
                type="range"
                min="10"
                max="60"
                step="5"
                value={targetFps}
                onChange={(e) => setTargetFps(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>10 FPS</span>
                <span>30 FPS (Standard)</span>
                <span>60 FPS</span>
              </div>
            </div>
          </div>
        </div>

        {/* AI Model Hyperparameters */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-white font-semibold text-sm border-b border-slate-800 pb-3">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span>AI Recognition & Detection Sensitivity</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-slate-300">
                  SFace Cosine Threshold
                </label>
                <span className="text-xs font-mono text-emerald-400 font-bold">{faceThreshold}</span>
              </div>
              <input
                type="range"
                min="0.25"
                max="0.60"
                step="0.01"
                value={faceThreshold}
                onChange={(e) => setFaceThreshold(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Standard: 0.363. Higher values require closer feature similarity to prevent imposters.
              </p>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-slate-300">
                  YuNet Face Min Confidence
                </label>
                <span className="text-xs font-mono text-emerald-400 font-bold">{detectorConfidence}</span>
              </div>
              <input
                type="range"
                min="0.40"
                max="0.95"
                step="0.05"
                value={detectorConfidence}
                onChange={(e) => setDetectorConfidence(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Minimum confidence to classify a region as a human face.
              </p>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-slate-300">
                  YOLOv8 Phone / Object Confidence
                </label>
                <span className="text-xs font-mono text-emerald-400 font-bold">{yoloConfidence}</span>
              </div>
              <input
                type="range"
                min="0.15"
                max="0.75"
                step="0.05"
                value={yoloConfidence}
                onChange={(e) => setYoloConfidence(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Threshold for identifying unauthorized phones, papers, and backpacks.
              </p>
            </div>
          </div>
        </div>

        {/* Academic Policy & Attendance Rules */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-white font-semibold text-sm border-b border-slate-800 pb-3">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span>Attendance Policy & Automation</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Late Entry Grace Period (Minutes)
              </label>
              <input
                type="number"
                min="0"
                max="45"
                value={lateThresholdMinutes}
                onChange={(e) => setLateThresholdMinutes(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Students recognized after this grace period will automatically be marked 'Late' instead of 'Present'.
              </p>
            </div>

            <div className="flex flex-col justify-center space-y-3 pt-2">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoMarkAbsent}
                  onChange={(e) => setAutoMarkAbsent(e.target.checked)}
                  className="rounded border-slate-700 text-emerald-600 focus:ring-emerald-500 w-4 h-4 bg-slate-950"
                />
                <span className="text-xs text-slate-200">
                  Auto-finalize remaining roster students as 'Absent' when period is stopped
                </span>
              </label>

              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={soundAlerts}
                  onChange={(e) => setSoundAlerts(e.target.checked)}
                  className="rounded border-slate-700 text-emerald-600 focus:ring-emerald-500 w-4 h-4 bg-slate-950"
                />
                <span className="text-xs text-slate-200">
                  Audio chime when high-priority malpractice alert is detected
                </span>
              </label>
            </div>
          </div>
        </div>

        {/* Database & Storage Status */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center space-x-2 text-white font-semibold text-sm border-b border-slate-800 pb-3">
            <HardDrive className="w-4 h-4 text-emerald-400" />
            <span>Storage & Biometric Persistence</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="bg-slate-950 border border-slate-800 p-3 rounded-lg">
              <div className="text-slate-400 font-medium">Database Engine</div>
              <div className="text-slate-200 font-bold mt-1">SQLite 3 (WAL Mode Enabled)</div>
              <div className="text-[10px] text-emerald-400 mt-1">Busy timeout: 30,000ms</div>
            </div>
            <div className="bg-slate-950 border border-slate-800 p-3 rounded-lg">
              <div className="text-slate-400 font-medium">Evidence Video Buffer</div>
              <div className="text-slate-200 font-bold mt-1">10-Second Circular Roll</div>
              <div className="text-[10px] text-slate-500 mt-1">H.264 mp4v encoding</div>
            </div>
            <div className="bg-slate-950 border border-slate-800 p-3 rounded-lg">
              <div className="text-slate-400 font-medium">Face Sample Quota</div>
              <div className="text-slate-200 font-bold mt-1">50 Samples / Enrollee</div>
              <div className="text-[10px] text-emerald-400 mt-1">Multi-angle YuNet quality checks</div>
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="submit"
            className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg text-sm shadow-lg shadow-emerald-900/30 transition flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Save Configuration</span>
          </button>
        </div>
      </form>
    </div>
  );
};
