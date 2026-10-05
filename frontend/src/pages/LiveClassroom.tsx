import React, { useEffect, useRef, useState, useCallback } from 'react';
import { 
  Play, Square, Video, AlertTriangle, CheckCircle, 
  Clock, ShieldAlert, Cpu, Sparkles 
} from 'lucide-react';
import { api } from '../services/api';
import { ClassItem, TimetablePeriod, TrackedPerson, DetectedObject, Student } from '../types';

export const LiveClassroom: React.FC = () => {
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [studentsRoster, setStudentsRoster] = useState<Student[]>([]);
  const [currentPeriod, setCurrentPeriod] = useState<TimetablePeriod | null>(null);
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [teacherConfirmed, setTeacherConfirmed] = useState<boolean>(false);
  const [teacherName, setTeacherName] = useState<string>('');
  const [sessionId, setSessionId] = useState<number | null>(null);

  // Live telemetry & counts
  const [fpsInfo, setFpsInfo] = useState({ e2e_fps: 0, det_fps: 0, rec_fps: 0, latency_ms: 0 });
  const [counts, setCounts] = useState({ total: 0, present: 0, absent: 0, percentage: 0 });
  const [trackedPersons, setTrackedPersons] = useState<TrackedPerson[]>([]);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [liveAlerts, setLiveAlerts] = useState<Array<{ id: string; name: string; type: string; time: string }>>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const isStreamingRef = useRef<boolean>(false);

  // Load Classes
  useEffect(() => {
    api.getClasses().then((res) => {
      setClasses(res.classes);
      if (res.classes.length > 0) {
        setSelectedClassId(res.classes[0].id);
      }
    });
  }, []);

  // When class changes, fetch timetable and roster
  useEffect(() => {
    if (!selectedClassId) return;

    api.getStudents(selectedClassId).then((res) => {
      setStudentsRoster(res.students);
    });

    api.getCurrentPeriod(selectedClassId).then((res) => {
      setCurrentPeriod(res.active_period);
      if (res.active_period?.teacher_name) {
        setTeacherName(res.active_period.teacher_name);
      }
    });

    api.getClassroomStatus(selectedClassId).then((res) => {
      setIsSessionActive(res.is_active);
      if (res.session) {
        setSessionId(res.session.session_id);
        setTeacherConfirmed(res.session.teacher_confirmed);
        if (res.session.teacher_name) setTeacherName(res.session.teacher_name);
      }
    });
  }, [selectedClassId]);

  // Start Camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      isStreamingRef.current = true;
      connectWebSocket();
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Unable to access webcam. Please ensure camera permissions are granted.');
    }
  };

  // Stop Camera
  const stopCamera = () => {
    isStreamingRef.current = false;
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
  };

  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/classroom`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Classroom WebSocket connected');
      startFrameStreaming();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'detections') {
          setFpsInfo({
            e2e_fps: data.fps?.e2e_fps || 0,
            det_fps: data.fps?.detection_fps || 0,
            rec_fps: data.fps?.recognition_fps || 0,
            latency_ms: data.fps?.latency_ms || 0,
          });
          setCounts(data.counts || { total: 0, present: 0, absent: 0, percentage: 0 });
          setTrackedPersons(data.tracked || []);
          setDetectedObjects(data.objects || []);

          if (data.teacher_confirmed) {
            setTeacherConfirmed(true);
            if (data.teacher_name) setTeacherName(data.teacher_name);
          }

          // Check for high-severity activities to append to live alerts
          data.tracked?.forEach((p: TrackedPerson) => {
            if (p.activity && p.activity !== 'NORMAL' && p.activity !== 'ATTENTIVE') {
              setLiveAlerts((prev) => {
                const nowStr = new Date().toLocaleTimeString();
                if (!prev.some((a) => a.name === p.name && a.type === p.activity)) {
                  return [{ id: `${Date.now()}-${p.track_id}`, name: p.name, type: p.activity, time: nowStr }, ...prev.slice(0, 7)];
                }
                return prev;
              });
            }
          });

          // Draw Overlays
          drawCanvasOverlays(data.tracked || [], data.objects || []);
        }
      } catch (e) {
        console.error('WS Parse Error:', e);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };
  }, [selectedClassId]);

  // Frame streaming loop
  const startFrameStreaming = () => {
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = 640;
    hiddenCanvas.height = 480;
    const hiddenCtx = hiddenCanvas.getContext('2d');

    let lastSend = 0;
    const targetInterval = 100; // send frame every ~100ms (10 FPS) to balance inference

    const processLoop = (time: number) => {
      if (!isStreamingRef.current) return;

      if (time - lastSend >= targetInterval && videoRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        if (videoRef.current.videoWidth > 0 && hiddenCtx) {
          hiddenCtx.drawImage(videoRef.current, 0, 0, 640, 480);
          const base64Data = hiddenCanvas.toDataURL('image/jpeg', 0.65).split(',')[1];
          wsRef.current.send(
            JSON.stringify({
              type: 'frame',
              image: base64Data,
              class_id: selectedClassId,
              session_id: sessionId,
            })
          );
          lastSend = time;
        }
      }

      animationFrameId.current = requestAnimationFrame(processLoop);
    };

    animationFrameId.current = requestAnimationFrame(processLoop);
  };

  // Draw Bounding Boxes and HUD directly onto Canvas
  const drawCanvasOverlays = (tracked: TrackedPerson[], objects: DetectedObject[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw YOLO Objects (Cell Phone, Laptop, Book)
    objects.forEach((obj) => {
      const [x, y, w, h] = obj.box;
      if (obj.label === 'cell phone') {
        ctx.strokeStyle = '#f43f5e'; // Rose / Red
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#f43f5e';
        ctx.fillRect(x, Math.max(0, y - 22), 120, 22);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(`PHONE (${Math.round(obj.conf * 100)}%)`, x + 4, Math.max(14, y - 6));
      } else if (obj.label === 'laptop' || obj.label === 'book') {
        ctx.strokeStyle = '#38bdf8'; // Sky blue
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#0284c7';
        ctx.fillRect(x, Math.max(0, y - 18), 70, 18);
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px sans-serif';
        ctx.fillText(obj.label.toUpperCase(), x + 4, Math.max(12, y - 4));
      }
    });

    // 2. Draw Tracked People & Face Status
    tracked.forEach((p) => {
      const [x, y, w, h] = p.box;
      let strokeColor = '#10b981'; // Green for verified
      let badgeBg = '#059669';

      if (p.status === 'TRACK_ASSOCIATED') {
        strokeColor = '#f59e0b'; // Amber for turned around / occluded
        badgeBg = '#d97706';
      } else if (p.status === 'UNKNOWN') {
        strokeColor = '#ef4444'; // Red for unknown / unallocated
        badgeBg = '#dc2626';
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);

      // Label background
      const labelText = `${p.name} ${p.usn ? `(${p.usn})` : ''}`;
      const statusText = `${p.status} • ${p.activity}`;

      ctx.fillStyle = badgeBg;
      ctx.fillRect(x, Math.max(0, y - 36), Math.max(160, ctx.measureText(labelText).width + 12), 34);

      // Text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(labelText, x + 6, Math.max(14, y - 19));

      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(statusText, x + 6, Math.max(26, y - 6));
    });
  };

  // Start Live Monitoring Session
  const handleStartSession = async () => {
    if (!selectedClassId) return;
    try {
      const res = await api.startClassroomSession(selectedClassId, currentPeriod?.subject, true);
      setIsSessionActive(true);
      setSessionId(res.session.session_id);
      if (res.session.teacher_name) setTeacherName(res.session.teacher_name);
      alert(`Classroom Session Started: ${res.message}`);
    } catch (err: any) {
      alert(`Failed to start session: ${err.message}`);
    }
  };

  // Stop Live Monitoring Session
  const handleStopSession = async () => {
    if (!selectedClassId) return;
    if (!confirm('Are you sure you want to stop the class session? Absent students will be finalized.')) return;

    try {
      const res = await api.stopClassroomSession(selectedClassId);
      setIsSessionActive(false);
      setSessionId(null);
      alert(`Classroom Session Finalized!\nPresent: ${res.summary.present_count}\nAbsent: ${res.summary.absent_count}`);
      // Refresh roster
      api.getStudents(selectedClassId).then((r) => setStudentsRoster(r.students));
    } catch (err: any) {
      alert(`Failed to stop session: ${err.message}`);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Controls Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="text-xs font-medium text-slate-400 block mb-1">Select Active Class</label>
            <select
              value={selectedClassId || ''}
              onChange={(e) => setSelectedClassId(Number(e.target.value))}
              className="bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
            >
              {classes.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.department} - Sem {c.semester})
                </option>
              ))}
            </select>
          </div>

          {/* Timetable Badge */}
          <div className="border-l border-slate-800 pl-4">
            <span className="text-xs font-medium text-slate-400 block mb-1">Timetable Verification</span>
            {currentPeriod ? (
              <div className="flex items-center space-x-2 text-xs bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-2.5 py-1.5 rounded-lg">
                <Clock className="w-3.5 h-3.5" />
                <span className="font-semibold">{currentPeriod.subject}</span>
                <span>({currentPeriod.start_time} - {currentPeriod.end_time})</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2 text-xs bg-amber-500/10 border border-amber-500/20 text-amber-400 px-2.5 py-1.5 rounded-lg">
                <Clock className="w-3.5 h-3.5" />
                <span>No class scheduled now (Override permitted)</span>
              </div>
            )}
          </div>

          {/* Teacher Verification Status */}
          <div className="border-l border-slate-800 pl-4 hidden md:block">
            <span className="text-xs font-medium text-slate-400 block mb-1">Teacher Presence</span>
            <div className="flex items-center space-x-1.5 text-xs">
              {teacherConfirmed ? (
                <span className="inline-flex items-center text-emerald-400 font-semibold bg-emerald-950/40 border border-emerald-800/40 px-2 py-1 rounded">
                  <CheckCircle className="w-3.5 h-3.5 mr-1" />
                  {teacherName || 'Confirmed Present'}
                </span>
              ) : (
                <span className="inline-flex items-center text-amber-400 bg-amber-950/40 border border-amber-800/40 px-2 py-1 rounded">
                  <AlertTriangle className="w-3.5 h-3.5 mr-1" />
                  Waiting for Teacher
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-3">
          {!isStreamingRef.current ? (
            <button
              onClick={startCamera}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm transition shadow-md shadow-blue-900/30"
            >
              <Video className="w-4 h-4" />
              <span>Open Camera</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="inline-flex items-center space-x-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm border border-slate-700 transition"
            >
              <span>Turn Off Camera</span>
            </button>
          )}

          {!isSessionActive ? (
            <button
              onClick={handleStartSession}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm transition shadow-md shadow-emerald-900/30"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>Start Class</span>
            </button>
          ) : (
            <button
              onClick={handleStopSession}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white font-semibold text-sm transition shadow-md shadow-red-900/30"
            >
              <Square className="w-4 h-4 fill-white" />
              <span>Stop Class</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Monitoring Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Live Video Feed & Canvas */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative aspect-[4/3] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center">
            {/* Background Video */}
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover"
              playsInline
              muted
            />

            {/* Foreground Overlay Canvas */}
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            />

            {/* Offline state placeholder */}
            {!isStreamingRef.current && (
              <div className="text-center p-8 z-10">
                <div className="w-16 h-16 rounded-full bg-slate-900/80 border border-slate-800 flex items-center justify-center mx-auto mb-4 text-slate-500">
                  <Video className="w-8 h-8" />
                </div>
                <h3 className="text-base font-semibold text-white">Camera Offline</h3>
                <p className="text-xs text-slate-400 max-w-sm mt-1">
                  Click "Open Camera" to activate real-time YOLOv8, SFace, and MediaPipe monitoring.
                </p>
                <button
                  onClick={startCamera}
                  className="mt-4 px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold text-sm transition"
                >
                  Start Webcam
                </button>
              </div>
            )}

            {/* Top Telemetry HUD */}
            {isStreamingRef.current && (
              <div className="absolute top-3 left-3 right-3 flex items-center justify-between bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800/80 text-[11px] text-slate-300 font-mono z-20">
                <div className="flex items-center space-x-3">
                  <span className="flex items-center text-emerald-400 font-semibold">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping mr-1.5" />
                    LIVE
                  </span>
                  <span>E2E: {fpsInfo.e2e_fps} FPS ({fpsInfo.latency_ms}ms)</span>
                  <span className="hidden sm:inline">Det: {fpsInfo.det_fps} FPS</span>
                  <span className="hidden sm:inline">Rec: {fpsInfo.rec_fps} FPS</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Cpu className="w-3.5 h-3.5 text-blue-400" />
                  <span>YOLOv8 + YuNet/SFace</span>
                </div>
              </div>
            )}
          </div>

          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-4 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-center">
              <span className="text-[10px] uppercase font-bold text-slate-400">Class Roster</span>
              <div className="text-xl font-bold text-white mt-0.5">{counts.total || studentsRoster.length}</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-center">
              <span className="text-[10px] uppercase font-bold text-emerald-400">Present</span>
              <div className="text-xl font-bold text-emerald-400 mt-0.5">{counts.present}</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-center">
              <span className="text-[10px] uppercase font-bold text-slate-400">Absent</span>
              <div className="text-xl font-bold text-slate-400 mt-0.5">{counts.absent}</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-center">
              <span className="text-[10px] uppercase font-bold text-emerald-400">Attendance</span>
              <div className="text-xl font-bold text-emerald-400 mt-0.5">{counts.percentage}%</div>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Live Roster & Malpractice Alerts Drawer */}
        <div className="space-y-4">
          {/* Real-Time Live Activity & Malpractice Feed */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-1.5">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                <span>Observable Behavior Stream</span>
              </h3>
              <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">
                {liveAlerts.length} events
              </span>
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {liveAlerts.length === 0 ? (
                <div className="py-6 text-center text-xs text-slate-500">
                  <CheckCircle className="w-6 h-6 mx-auto mb-1 text-emerald-500/40" />
                  No disengagement or phone events detected
                </div>
              ) : (
                liveAlerts.map((ev) => (
                  <div
                    key={ev.id}
                    className="p-2 rounded-lg bg-slate-800/80 border border-slate-700/60 flex items-center justify-between text-xs"
                  >
                    <div>
                      <div className="font-semibold text-slate-200">{ev.name}</div>
                      <div className="text-amber-400 text-[11px] font-mono">{ev.type}</div>
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">{ev.time}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Student Class Roster with Real Attendance State */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Student Attendance Status
              </h3>
              <span className="text-[11px] text-emerald-400 font-semibold font-mono">
                {counts.present}/{studentsRoster.length} Verified
              </span>
            </div>

            <div className="space-y-1.5 max-h-80 overflow-y-auto pr-1 divide-y divide-slate-800/60">
              {studentsRoster.map((st) => {
                // Check if currently tracked or marked present
                const isPresent = trackedPersons.some(
                  (p) => p.name === st.name && p.attendance_marked
                );

                return (
                  <div key={st.id} className="pt-2 flex items-center justify-between text-xs">
                    <div>
                      <div className="font-medium text-slate-200">{st.name}</div>
                      <div className="text-[10px] text-slate-400 font-mono">Roll: {st.roll_no} • {st.usn}</div>
                    </div>
                    <div>
                      {isPresent ? (
                        <span className="inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          PRESENT
                        </span>
                      ) : (
                        <span className="inline-flex items-center text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                          ABSENT
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
