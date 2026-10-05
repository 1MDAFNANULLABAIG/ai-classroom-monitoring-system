import React, { useEffect, useRef, useState } from 'react';
import { Camera, CheckCircle2, RefreshCw, X, Check } from 'lucide-react';
import { Teacher } from '../types';
import { api } from '../services/api';

interface TeacherEnrollmentModalProps {
  teacher: Teacher;
  onClose: () => void;
  onEnrollmentComplete: () => void;
}

export const TeacherEnrollmentModal: React.FC<TeacherEnrollmentModalProps> = ({
  teacher,
  onClose,
  onEnrollmentComplete,
}) => {
  const [currentCount, setCurrentCount] = useState<number>(teacher.sample_count || 0);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [autoBurst, setAutoBurst] = useState<boolean>(true);
  const [qualityFeedback, setQualityFeedback] = useState<{
    pose?: string;
    score?: number;
    message?: string;
    passed?: boolean;
  }>({});
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isComplete, setIsComplete] = useState<boolean>(teacher.face_enrolled || teacher.sample_count >= 50);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoBurstInterval = useRef<any>(null);

  const TARGET_SAMPLES = 50;

  useEffect(() => {
    startWebcam();
    return () => {
      stopWebcam();
    };
  }, []);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 480, height: 480, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error('Camera open error:', err);
      alert('Unable to access webcam. Please check permissions.');
    }
  };

  const stopWebcam = () => {
    if (autoBurstInterval.current) {
      clearInterval(autoBurstInterval.current);
      autoBurstInterval.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  const captureFrame = async () => {
    if (!videoRef.current || isCapturing || isComplete) return;

    const video = videoRef.current;
    if (video.videoWidth === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const base64 = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];

    setIsCapturing(true);
    try {
      const res = await api.enrollTeacherFrame(teacher.id, base64);
      if (res.success) {
        setCurrentCount(res.count);
        setQualityFeedback({
          pose: res.pose,
          score: Math.round(res.quality_score * 100),
          message: `Sample accepted! (${res.pose})`,
          passed: true,
        });

        if (res.count >= TARGET_SAMPLES || res.auto_trained) {
          setIsComplete(true);
          if (autoBurstInterval.current) {
            clearInterval(autoBurstInterval.current);
            autoBurstInterval.current = null;
          }
          onEnrollmentComplete();
        }
      }
    } catch (err: any) {
      setQualityFeedback({
        message: err.message,
        passed: false,
      });
    } finally {
      setIsCapturing(false);
    }
  };

  useEffect(() => {
    if (autoBurst && !isComplete) {
      autoBurstInterval.current = setInterval(() => {
        captureFrame();
      }, 350);
    } else {
      if (autoBurstInterval.current) {
        clearInterval(autoBurstInterval.current);
        autoBurstInterval.current = null;
      }
    }
    return () => {
      if (autoBurstInterval.current) clearInterval(autoBurstInterval.current);
    };
  }, [autoBurst, isComplete, currentCount]);

  const handleManualTrain = async () => {
    setIsTraining(true);
    try {
      await api.trainTeacherModel(teacher.id);
      setIsComplete(true);
      alert('Teacher face model successfully trained!');
      onEnrollmentComplete();
    } catch (err: any) {
      alert(`Training error: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const progressPercent = Math.min(100, Math.round((currentCount / TARGET_SAMPLES) * 100));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-xl overflow-hidden shadow-2xl">
        <div className="p-4 sm:p-5 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <Camera className="w-5 h-5 text-purple-400" />
              <span>Teacher Face Enrollment: {teacher.name}</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Teacher Code: {teacher.teacher_code} • Email: {teacher.email || 'N/A'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <div className="relative aspect-square max-w-[340px] mx-auto bg-slate-950 rounded-2xl overflow-hidden border-2 border-slate-800 shadow-inner flex items-center justify-center">
            <video
              ref={videoRef}
              className="w-full h-full object-cover transform -scale-x-100"
              playsInline
              muted
            />

            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <div
                className={`w-48 h-60 rounded-[50%] border-2 transition-colors ${
                  qualityFeedback.passed ? 'border-purple-400 shadow-[0_0_20px_rgba(192,132,252,0.3)]' : 'border-slate-500/60'
                }`}
              />
            </div>

            {qualityFeedback.message && (
              <div
                className={`absolute bottom-3 left-3 right-3 text-center px-3 py-1.5 rounded-lg text-xs font-semibold backdrop-blur-md ${
                  qualityFeedback.passed
                    ? 'bg-purple-950/80 text-purple-300 border border-purple-800/60'
                    : 'bg-red-950/80 text-red-300 border border-red-800/60'
                }`}
              >
                {qualityFeedback.message}
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
              <span className="text-slate-300">
                Enrollment Progress: {currentCount} / {TARGET_SAMPLES} Samples
              </span>
              <span className="text-purple-400 font-mono">{progressPercent}%</span>
            </div>
            <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700/60">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-indigo-400 transition-all duration-300 rounded-full"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>

          {isComplete && (
            <div className="p-3 rounded-xl bg-purple-950/40 border border-purple-800/60 flex items-center space-x-3 text-purple-300 text-xs">
              <div className="p-1.5 rounded-full bg-purple-500/20 text-purple-400">
                <Check className="w-4 h-4" />
              </div>
              <div>
                <span className="font-bold block">Teacher Enrolled Successfully!</span>
                <span>The system can now identify this teacher before unlocking student attendance.</span>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoBurst}
                disabled={isComplete}
                onChange={(e) => setAutoBurst(e.target.checked)}
                className="w-4 h-4 rounded text-purple-500 bg-slate-800 border-slate-700 focus:ring-0"
              />
              <span>Auto-Burst Capture</span>
            </label>

            <div className="flex items-center space-x-2">
              {!isComplete ? (
                <>
                  <button
                    onClick={captureFrame}
                    disabled={isCapturing || autoBurst}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 disabled:opacity-50"
                  >
                    Capture Single
                  </button>
                  <button
                    onClick={handleManualTrain}
                    disabled={currentCount < 10 || isTraining}
                    className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-xs font-semibold disabled:opacity-50 transition"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
                    <span>Train Model</span>
                  </button>
                </>
              ) : (
                <button
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg bg-purple-500 hover:bg-purple-600 text-slate-950 text-xs font-bold transition"
                >
                  Done
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
