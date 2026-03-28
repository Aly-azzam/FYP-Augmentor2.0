import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { toast } from 'sonner';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Volume2,
  VolumeX,
  Upload,
  MousePointer,
  ArrowUpRight,
  Minus,
  Square,
  Circle,
  PenTool,
  Triangle,
  Ruler,
  Crosshair,
  Trash2,
  Plus,
  RotateCcw,
  Flag,
  Video,
  Sparkles,
} from 'lucide-react';
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from '../components/ui/tabs';
import { Switch } from '../components/ui/switch';
import { Slider } from '../components/ui/slider';
import { Progress } from '../components/ui/progress';
import {
  useCourseStore,
  useVideoSyncStore,
  useDrawingStore,
  useEvaluationStore,
  useUIStore,
  useTimerStore,
} from '../store';
import { courses, getClipsForCourse } from '../services/mock/courses';
import { getEvaluationResult, startEvaluation as startEvaluationApi } from '../api/evaluationApi';
import { formatTime } from '../utils/helpers';
import type { DrawingTool } from '../types';

// ── Constants ────────────────────────────────────────────────────────────────

const PIPELINE_STAGES = [
  'Validation',
  'Pose Analysis',
  'DTW Alignment',
  'Metric Computation',
  'VLM Explanation',
];

const DRAWING_TOOLS: {
  id: DrawingTool;
  label: string;
  icon: React.ReactNode;
}[] = [
  { id: 'select', label: 'Select', icon: <MousePointer size={16} /> },
  { id: 'arrow', label: 'Arrow', icon: <ArrowUpRight size={16} /> },
  { id: 'line', label: 'Line', icon: <Minus size={16} /> },
  { id: 'rectangle', label: 'Rect', icon: <Square size={16} /> },
  { id: 'circle', label: 'Circle', icon: <Circle size={16} /> },
  { id: 'pen', label: 'Pen', icon: <PenTool size={16} /> },
  { id: 'angle', label: 'Angle', icon: <Triangle size={16} /> },
  { id: 'calibrate', label: 'Calibrate', icon: <Ruler size={16} /> },
  { id: 'track', label: 'Track', icon: <Crosshair size={16} /> },
];

const COLOR_SWATCHES = [
  '#2563EB',
  '#EF4444',
  '#10B981',
  '#F59E0B',
  '#8B5CF6',
  '#EC4899',
];

const PLAYBACK_RATES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

const scoreColor = (score: number): string => {
  if (score >= 90) return '#22c55e';
  if (score >= 70) return '#3b82f6';
  if (score >= 50) return '#eab308';
  return '#ef4444';
};

const scoreLabel = (score: number): string => {
  if (score >= 90) return 'Excellent';
  if (score >= 80) return 'Very Good';
  if (score >= 70) return 'Good';
  if (score >= 50) return 'Fair';
  return 'Needs Improvement';
};

/** File inputs on Windows often omit MIME type; allow .mp4 / .mov by extension. */
function isValidPracticeVideo(file: File): boolean {
  const t = (file.type || '').toLowerCase();
  if (t === 'video/mp4' || t === 'video/quicktime') return true;
  const ext = file.name.toLowerCase().match(/\.([^.]+)$/)?.[1];
  return ext === 'mp4' || ext === 'mov';
}

const TOUR_STEPS = [
  {
    title: 'Welcome to Compare Studio',
    description:
      "Compare your practice videos side-by-side with expert demonstrations. Let's take a quick tour!",
  },
  {
    title: 'Expert Video',
    description:
      "The left panel plays the instructor's demonstration. Use the timeline and playback controls to scrub through.",
  },
  {
    title: 'Upload Your Practice',
    description:
      'Drag and drop your practice video into the right panel, or click to browse. MP4 and MOV up to 2 minutes.',
  },
  {
    title: 'Drawing Tools',
    description:
      'Annotate the video with arrows, lines, angles, and more. Trace paths and measure distances to analyse technique.',
  },
  {
    title: 'AI Evaluation',
    description:
      'Run the AI pipeline to get a detailed score, metric breakdown, and natural-language feedback.',
  },
];

// ── Component ────────────────────────────────────────────────────────────────

export default function CompareStudio() {
  // ── Store hooks ──────────────────────────────────────────────────────────

  const [searchParams] = useSearchParams();

  const { selectedCourse, selectedClip, userVideo, setUserVideo, setSelectedCourse, setSelectedClip } =
    useCourseStore();

  const {
    isSynced,
    playbackRate,
    isPlaying,
    offset,
    setIsSynced,
    setPlaybackRate,
    setIsPlaying,
    setOffset,
  } = useVideoSyncStore();

  const {
    activeTool,
    toolColor,
    toolThickness,
    showAllFrames,
    setActiveTool,
    setToolColor,
    setToolThickness,
    setShowAllFrames,
    clearTrackedPoints,
    clearMeasurements,
  } = useDrawingStore();

  const {
    isEvaluating,
    evaluationStep,
    evaluationProgress,
    startEvaluation,
    setEvaluationStep,
    setEvaluationProgress,
    resetEvaluation,
  } = useEvaluationStore();

  const {
    showGuidedTour,
    tourStep,
    setShowGuidedTour,
    setRobotMessage,
    nextTourStep,
    prevTourStep,
  } = useUIStore();

  const { timers, addTimer, startTimer, stopTimer, resetTimer, addTimestamp, updateElapsed } =
    useTimerStore();

  // ── Local state ──────────────────────────────────────────────────────────

  const [userVideoUrl, setUserVideoUrl] = useState<string | null>(null);
  const [expertVideoUrl, setExpertVideoUrl] = useState<string | null>(null);
  const [expertVideoError, setExpertVideoError] = useState<string | null>(null);
  const [expertCurrentTime, setExpertCurrentTime] = useState(0);
  const [expertDuration, setExpertDuration] = useState(0);
  const [learnerCurrentTime, setLearnerCurrentTime] = useState(0);
  const [learnerDuration, setLearnerDuration] = useState(0);
  const [expertMuted, setExpertMuted] = useState(false);
  const [learnerMuted, setLearnerMuted] = useState(false);
  const [apiEvaluationResult, setApiEvaluationResult] = useState<any | null>(null);

  // ── Refs ──────────────────────────────────────────────────────────────────

  const expertVideoRef = useRef<HTMLVideoElement>(null);
  const learnerVideoRef = useRef<HTMLVideoElement>(null);
  const expertCanvasRef = useRef<HTMLCanvasElement>(null);
  const learnerFileInputRef = useRef<HTMLInputElement>(null);
  /** Tracks the active learner blob URL so we only revoke on replace / unmount (avoids Strict Mode double-revoke). */
  const learnerBlobUrlRef = useRef<string | null>(null);
  const timersRef = useRef(timers);
  timersRef.current = timers;

  // ── Derived data ─────────────────────────────────────────────────────────

  const course = courses.find((c) => c.id === selectedCourse);
  const clips = selectedCourse ? getClipsForCourse(selectedCourse) : [];
  const clip = clips.find((c) => c.id === selectedClip);
  const requestedCourseId = searchParams.get('courseId');
  const requestedClipId = searchParams.get('clipId');

  // ── Effects ──────────────────────────────────────────────────────────────

  useEffect(() => {
    if (requestedCourseId && requestedCourseId !== selectedCourse) {
      setSelectedCourse(requestedCourseId);
    }

    if (requestedClipId && requestedClipId !== selectedClip) {
      setSelectedClip(requestedClipId);
    }
  }, [
    requestedCourseId,
    requestedClipId,
    selectedCourse,
    selectedClip,
    setSelectedCourse,
    setSelectedClip,
  ]);

  useEffect(() => {
    setRobotMessage(
      clip
        ? `Practicing "${clip.title}". Upload your video and compare with the expert!`
        : 'Choose an expert video first, then start the comparison from the course page.',
    );
    return () => setRobotMessage(null);
  }, [clip, setRobotMessage]);

  useEffect(() => {
    if (!clip) {
      setExpertVideoUrl(null);
      setExpertVideoError(null);
      setExpertCurrentTime(0);
      setExpertDuration(0);
    }
  }, [clip]);

  useEffect(() => {
    let isCancelled = false;

    const loadSelectedExpertVideo = async () => {
      if (!clip) {
        return;
      }

      if (clip.expertVideoUrl) {
        setExpertVideoError(null);
        setExpertVideoUrl(clip.expertVideoUrl);
        return;
      }

      try {
        if (!isCancelled) {
          setExpertVideoUrl(null);
          setExpertVideoError(
            'This selected clip does not have a real expert video linked yet.',
          );
        }
      } catch {
        if (!isCancelled) {
          setExpertVideoUrl(null);
          setExpertVideoError(
            'This selected clip does not have a real expert video linked yet.',
          );
        }
      }
    };

    void loadSelectedExpertVideo();

    return () => {
      isCancelled = true;
    };
  }, [clip]);

  // Timer tick
  useEffect(() => {
    const id = setInterval(() => {
      timersRef.current.forEach((t) => {
        if (t.isRunning) updateElapsed(t.id, t.elapsed + 0.1);
      });
    }, 100);
    return () => clearInterval(id);
  }, [updateElapsed]);

  // Sync play / pause to <video> elements
  useEffect(() => {
    if (expertVideoRef.current) {
      if (isPlaying) expertVideoRef.current.play().catch(() => {});
      else expertVideoRef.current.pause();
    }
    if (learnerVideoRef.current && userVideoUrl) {
      if (isPlaying) learnerVideoRef.current.play().catch(() => {});
      else learnerVideoRef.current.pause();
    }
  }, [isPlaying, userVideoUrl]);

  // Sync playback rate
  useEffect(() => {
    if (expertVideoRef.current) expertVideoRef.current.playbackRate = playbackRate;
    if (learnerVideoRef.current) learnerVideoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  // Mute sync
  useEffect(() => {
    if (expertVideoRef.current) expertVideoRef.current.muted = expertMuted;
  }, [expertMuted]);
  useEffect(() => {
    if (learnerVideoRef.current) learnerVideoRef.current.muted = learnerMuted;
  }, [learnerMuted]);

  // Revoke learner blob URL only when leaving the page (not on every URL change — that could revoke the new blob).
  useEffect(() => {
    return () => {
      const u = learnerBlobUrlRef.current;
      if (u) {
        URL.revokeObjectURL(u);
        learnerBlobUrlRef.current = null;
      }
    };
  }, []);

  // ── Video upload ─────────────────────────────────────────────────────────

  const handlePracticeVideoFile = useCallback(
    (file: File) => {
      if (!isValidPracticeVideo(file)) {
        toast.error('Please upload an MP4 or MOV file');
        return;
      }

      const url = URL.createObjectURL(file);
      const probe = document.createElement('video');
      probe.preload = 'metadata';
      probe.onerror = () => {
        URL.revokeObjectURL(url);
        toast.error('Could not read this video file');
      };
      probe.onloadedmetadata = () => {
        if (probe.duration > 120) {
          toast.error('Video must be 2 minutes or shorter');
          URL.revokeObjectURL(url);
          return;
        }
        const prev = learnerBlobUrlRef.current;
        if (prev && prev !== url) {
          URL.revokeObjectURL(prev);
        }
        learnerBlobUrlRef.current = url;
        setUserVideoUrl(url);
        setUserVideo(file);
        setLearnerCurrentTime(0);
        setLearnerDuration(0);
        setApiEvaluationResult(null);
        resetEvaluation();
        toast.success('Practice video ready');
      };
      probe.src = url;
    },
    [setUserVideo, resetEvaluation],
  );

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      handlePracticeVideoFile(file);
    },
    [handlePracticeVideoFile],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/mp4': ['.mp4'], 'video/quicktime': ['.mov'] },
    maxFiles: 1,
  });

  // ── Evaluation pipeline ──────────────────────────────────────────────────

  const runEvaluation = useCallback(async () => {
    if (!selectedCourse || !selectedClip || !userVideo) return;
    try {
      setApiEvaluationResult(null);
      startEvaluation();
      setEvaluationStep(0);
      setEvaluationProgress(20);

      const formData = new FormData();
      formData.append('file', userVideo);
      formData.append('course_id', selectedCourse);
      formData.append('clip_id', selectedClip);
      formData.append('filename', userVideo.name);

      const started = await startEvaluationApi(formData);

      if (started?.status === 'out_of_context') {
        resetEvaluation();
        setApiEvaluationResult({
          score: 0,
          status: 'out_of_context',
          message: started.message,
          gate_reasons: started.gate_reasons || [],
          metrics: null,
          explanation: null,
          key_error_moments: [],
        });
        toast.error('Video rejected: does not match the expert task');
        return;
      }

      const evaluationId =
        typeof started === 'string'
          ? started
          : started?.evaluation_id || started?.id;

      if (!evaluationId) {
        throw new Error('Missing evaluation_id from backend response');
      }

      setEvaluationStep(2);
      setEvaluationProgress(70);
      const result = await getEvaluationResult(evaluationId);
      setEvaluationStep(4);
      setEvaluationProgress(100);
      setApiEvaluationResult(result);
      resetEvaluation();
      console.log('CompareStudio evaluation response:', result);
      toast.success(`Evaluation complete! Score: ${result.score}/100`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Evaluation failed';
      toast.error(message);
      resetEvaluation();
    }
  }, [
    selectedCourse,
    selectedClip,
    userVideo,
    startEvaluation,
    setEvaluationStep,
    setEvaluationProgress,
    resetEvaluation,
  ]);

  // ── Video helpers ────────────────────────────────────────────────────────

  const togglePlay = () => setIsPlaying(!isPlaying);

  const seekVideo = (
    e: React.MouseEvent<HTMLDivElement>,
    ref: React.RefObject<HTMLVideoElement | null>,
    duration: number,
  ) => {
    if (!ref.current || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    ref.current.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
  };

  const skipFrames = (
    ref: React.RefObject<HTMLVideoElement | null>,
    delta: number,
  ) => {
    if (ref.current)
      ref.current.currentTime = Math.max(0, ref.current.currentTime + delta);
  };

  const handleClearAll = () => {
    clearTrackedPoints();
    clearMeasurements();
    toast.info('All annotations cleared');
  };

  const handleAddTimer = () => {
    addTimer({
      id: `timer-${Date.now()}`,
      name: `Timer ${timers.length + 1}`,
      elapsed: 0,
      isRunning: false,
      timestamps: [],
    });
  };

  // ── Render helpers ───────────────────────────────────────────────────────

  const renderTimeline = (
    currentTime: number,
    duration: number,
    onClick: React.MouseEventHandler<HTMLDivElement>,
  ) => {
    const pct = duration ? (currentTime / duration) * 100 : 0;
    return (
      <div
        className="timeline"
        style={{ marginBottom: 'var(--space-sm)' }}
        onClick={onClick}
      >
        <div className="timeline-progress" style={{ width: `${pct}%` }} />
        <div className="timeline-handle" style={{ left: `${pct}%` }} />
      </div>
    );
  };

  const renderControls = (
    ref: React.RefObject<HTMLVideoElement | null>,
    currentTime: number,
    duration: number,
    muted: boolean,
    toggleMute: () => void,
  ) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-sm)',
        paddingTop: 'var(--space-xs)',
      }}
    >
      <button className="btn btn-ghost" onClick={() => skipFrames(ref, -1 / 30)}>
        <SkipBack size={16} />
      </button>
      <button
        className="btn btn-primary"
        style={{ borderRadius: '50%', width: 36, height: 36, padding: 0 }}
        onClick={togglePlay}
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
      </button>
      <button className="btn btn-ghost" onClick={() => skipFrames(ref, 1 / 30)}>
        <SkipForward size={16} />
      </button>

      <span
        className="text-small"
        style={{
          color: 'var(--text-secondary)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>

      <div style={{ marginLeft: 'auto' }}>
        <button className="btn btn-ghost" onClick={toggleMute}>
          {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
      </div>
    </div>
  );

  // ── JSX ──────────────────────────────────────────────────────────────────

  return (
    <div style={{ padding: 'var(--space-lg)', maxWidth: 1600, margin: '0 auto' }}>
      {/* ─── Sync Controls Bar ─────────────────────────────────────────── */}
      <motion.div
        className="glass"
        style={{
          borderRadius: 'var(--radius-lg)',
          padding: 'var(--space-sm) var(--space-lg)',
          marginBottom: 'var(--space-lg)',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-lg)',
          flexWrap: 'wrap',
        }}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <span className="label">Sync</span>
          <Switch checked={isSynced} onCheckedChange={setIsSynced} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <span className="label">Playback Rate</span>
          <select
            className="input"
            style={{ width: 80 }}
            value={playbackRate}
            onChange={(e) => setPlaybackRate(Number(e.target.value))}
          >
            {PLAYBACK_RATES.map((r) => (
              <option key={r} value={r}>
                {r}x
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <span className="label">Offset</span>
          <input
            className="input"
            type="number"
            step={0.1}
            style={{ width: 80 }}
            value={offset}
            onChange={(e) => setOffset(Number(e.target.value))}
          />
          <span className="text-small" style={{ color: 'var(--text-muted)' }}>
            sec
          </span>
        </div>

        <button className="btn btn-secondary" onClick={() => setOffset(0)}>
          Align Start
        </button>
      </motion.div>

      {/* ─── Main Grid: Expert | Learner | Sidebar ─────────────────────── */}
      <div
        className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_320px]"
        style={{ gap: 'var(--space-lg)' }}
      >
        {/* ── Expert Video ──────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <span
            className="label"
            style={{ display: 'block', marginBottom: 'var(--space-sm)' }}
          >
            Expert Video
          </span>

          {clip && expertVideoUrl ? (
            <>
              <div
                className="video-container"
                style={{
                  aspectRatio: '16/9',
                  marginBottom: 'var(--space-sm)',
                }}
              >
                <video
                  ref={expertVideoRef}
                  key={expertVideoUrl ?? 'missing-expert-video'}
                  src={expertVideoUrl ?? undefined}
                  poster={clip.thumbnail}
                  preload="metadata"
                  playsInline
                  onTimeUpdate={() => {
                    if (expertVideoRef.current)
                      setExpertCurrentTime(expertVideoRef.current.currentTime);
                  }}
                  onLoadedMetadata={() => {
                    if (expertVideoRef.current)
                      setExpertDuration(expertVideoRef.current.duration);
                  }}
                  style={{ background: 'var(--bg-tertiary)' }}
                />
                <canvas
                  ref={expertCanvasRef}
                  className={`canvas-overlay${activeTool !== 'select' ? ' interactive' : ''}`}
                />
              </div>

              {renderTimeline(expertCurrentTime, expertDuration, (e) =>
                seekVideo(e, expertVideoRef, expertDuration),
              )}
              {renderControls(
                expertVideoRef,
                expertCurrentTime,
                expertDuration,
                expertMuted,
                () => setExpertMuted(!expertMuted),
              )}
              {expertVideoError && (
                <p
                  className="text-small"
                  style={{ color: 'var(--danger, #dc2626)', marginTop: 'var(--space-sm)' }}
                >
                  {expertVideoError}
                </p>
              )}
            </>
          ) : (
            <div
              className="video-container flex-center"
              style={{
                aspectRatio: '16/9',
                background: 'var(--bg-tertiary)',
              }}
            >
              <div className="empty-state" style={{ padding: 'var(--space-xl)' }}>
                <Video
                  size={48}
                  style={{
                    color: 'var(--text-muted)',
                    margin: '0 auto var(--space-md)',
                  }}
                />
                <p className="empty-state-title">
                  {clip ? 'Expert Video Unavailable' : 'Choose An Expert Video'}
                </p>
                <p className="empty-state-description">
                  {clip
                    ? (expertVideoError ?? 'Start the backend and make sure the selected expert video is available.')
                    : 'Go to the course page, choose the expert video you want, then press start comparison.'}
                </p>
              </div>
            </div>
          )}
        </motion.div>

        {/* ── Learner Video / Upload ────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 'var(--space-sm)',
              marginBottom: 'var(--space-sm)',
            }}
          >
            <span className="label" style={{ margin: 0 }}>
              Your Practice
            </span>
            {userVideoUrl ? (
              <>
                <input
                  ref={learnerFileInputRef}
                  type="file"
                  accept="video/mp4,video/quicktime,.mp4,.mov"
                  style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }}
                  aria-hidden
                  tabIndex={-1}
                  onChange={(e) => {
                    const next = e.target.files?.[0];
                    e.target.value = '';
                    if (next) handlePracticeVideoFile(next);
                  }}
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  aria-label="Choose a different practice video"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 'var(--space-xs)',
                  }}
                  onClick={() => learnerFileInputRef.current?.click()}
                >
                  <Upload size={14} />
                  Change video
                </button>
              </>
            ) : null}
          </div>

          {userVideoUrl ? (
            <>
              <div
                className="video-container"
                style={{
                  aspectRatio: '16/9',
                  marginBottom: 'var(--space-sm)',
                }}
              >
                <video
                  key={userVideoUrl}
                  ref={learnerVideoRef}
                  src={userVideoUrl}
                  onTimeUpdate={() => {
                    if (learnerVideoRef.current)
                      setLearnerCurrentTime(learnerVideoRef.current.currentTime);
                  }}
                  onLoadedMetadata={() => {
                    if (learnerVideoRef.current)
                      setLearnerDuration(learnerVideoRef.current.duration);
                  }}
                />
              </div>

              {renderTimeline(learnerCurrentTime, learnerDuration, (e) =>
                seekVideo(e, learnerVideoRef, learnerDuration),
              )}
              {renderControls(
                learnerVideoRef,
                learnerCurrentTime,
                learnerDuration,
                learnerMuted,
                () => setLearnerMuted(!learnerMuted),
              )}
            </>
          ) : (
            <div
              {...getRootProps()}
              className="video-container flex-center"
              style={{
                aspectRatio: '16/9',
                background: 'var(--bg-tertiary)',
                border: '2px dashed',
                borderColor: isDragActive
                  ? 'var(--accent-primary)'
                  : 'var(--border-default)',
                cursor: 'pointer',
                transition: 'border-color var(--transition-fast)',
              }}
            >
              <input {...getInputProps()} />
              <div style={{ textAlign: 'center', padding: 'var(--space-xl)' }}>
                <Upload
                  size={48}
                  style={{
                    color: isDragActive
                      ? 'var(--accent-primary)'
                      : 'var(--text-muted)',
                    margin: '0 auto var(--space-md)',
                  }}
                />
                <p className="heading-4" style={{ marginBottom: 'var(--space-xs)' }}>
                  {isDragActive
                    ? 'Drop your video here'
                    : 'Drag & drop your practice video'}
                </p>
                <p
                  className="text-small"
                  style={{ color: 'var(--text-muted)' }}
                >
                  MP4 or MOV, max 2 minutes
                </p>
              </div>
            </div>
          )}
        </motion.div>

        {/* ── Sidebar: Tools / Evaluate / Timers ───────────────────────── */}
        <motion.div
          className="glass"
          style={{
            borderRadius: 'var(--radius-lg)',
            padding: 'var(--space-md)',
            maxHeight: 'calc(100vh - 180px)',
            overflowY: 'auto',
          }}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Tabs defaultValue="tools">
            <TabsList>
              <TabsTrigger value="tools">Tools</TabsTrigger>
              <TabsTrigger value="evaluate">Evaluate</TabsTrigger>
              <TabsTrigger value="timers">Timers</TabsTrigger>
            </TabsList>

            {/* ── Tab: Tools ───────────────────────────────────────────── */}
            <TabsContent value="tools">
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 'var(--space-md)',
                }}
              >
                {/* Tool grid */}
                <div>
                  <span
                    className="label"
                    style={{
                      display: 'block',
                      marginBottom: 'var(--space-sm)',
                    }}
                  >
                    Tools
                  </span>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: 'var(--space-xs)',
                    }}
                  >
                    {DRAWING_TOOLS.map((tool) => (
                      <button
                        key={tool.id}
                        className={`btn ${activeTool === tool.id ? 'btn-primary' : 'btn-secondary'}`}
                        style={{
                          flexDirection: 'column',
                          padding: 'var(--space-sm)',
                          fontSize: '0.7rem',
                          gap: '0.25rem',
                        }}
                        onClick={() => setActiveTool(tool.id)}
                      >
                        {tool.icon}
                        {tool.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Color picker */}
                <div>
                  <span
                    className="label"
                    style={{
                      display: 'block',
                      marginBottom: 'var(--space-sm)',
                    }}
                  >
                    Color
                  </span>
                  <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                    {COLOR_SWATCHES.map((color) => (
                      <button
                        key={color}
                        aria-label={`Color ${color}`}
                        onClick={() => setToolColor(color)}
                        style={{
                          width: 28,
                          height: 28,
                          borderRadius: '50%',
                          background: color,
                          border:
                            toolColor === color
                              ? '2px solid var(--text-primary)'
                              : '2px solid transparent',
                          cursor: 'pointer',
                          transition: 'transform var(--transition-fast)',
                          transform:
                            toolColor === color ? 'scale(1.15)' : 'scale(1)',
                          outline: 'none',
                        }}
                      />
                    ))}
                  </div>
                </div>

                {/* Thickness */}
                <div>
                  <div
                    className="flex-between"
                    style={{ marginBottom: 'var(--space-sm)' }}
                  >
                    <span className="label">Thickness</span>
                    <span
                      className="text-small"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {toolThickness}px
                    </span>
                  </div>
                  <Slider
                    value={[toolThickness]}
                    min={1}
                    max={10}
                    step={1}
                    onValueChange={([v]) => setToolThickness(v)}
                  />
                </div>

                {/* Show all frames */}
                <div className="flex-between">
                  <span className="text-small">Show on all frames</span>
                  <Switch
                    checked={showAllFrames}
                    onCheckedChange={setShowAllFrames}
                  />
                </div>

                <div className="divider" />

                {/* Clear all */}
                <button
                  className="btn btn-secondary"
                  style={{ width: '100%' }}
                  onClick={handleClearAll}
                >
                  <Trash2 size={14} />
                  Clear All
                </button>
              </div>
            </TabsContent>

            {/* ── Tab: Evaluate ─────────────────────────────────────────── */}
            <TabsContent value="evaluate">
              {/* Idle */}
              {!isEvaluating && !apiEvaluationResult && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-lg)',
                    padding: 'var(--space-lg) 0',
                  }}
                >
                  <Sparkles
                    size={48}
                    style={{ color: 'var(--text-muted)' }}
                  />
                  <p
                    className="text-body"
                    style={{ textAlign: 'center' }}
                  >
                    {userVideo
                      ? 'Ready to evaluate your practice!'
                      : 'Upload a practice video to get started.'}
                  </p>
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%' }}
                    disabled={!userVideo || !selectedClip}
                    onClick={runEvaluation}
                  >
                    Run Evaluation
                  </button>
                </div>
              )}

              {/* Evaluating */}
              {isEvaluating && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-md)',
                    padding: 'var(--space-lg) 0',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 'var(--space-sm)',
                    }}
                  >
                    {PIPELINE_STAGES.map((stage, i) => (
                      <div
                        key={stage}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 'var(--space-sm)',
                        }}
                      >
                        <div
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            background:
                              i < evaluationStep
                                ? 'var(--success)'
                                : i === evaluationStep
                                  ? 'var(--accent-primary)'
                                  : 'var(--bg-tertiary)',
                            transition: 'background var(--transition-fast)',
                          }}
                        />
                        <span
                          className="text-small"
                          style={{
                            color:
                              i <= evaluationStep
                                ? 'var(--text-primary)'
                                : 'var(--text-muted)',
                            fontWeight: i === evaluationStep ? 600 : 400,
                          }}
                        >
                          {stage}
                        </span>
                        {i < evaluationStep && (
                          <span
                            style={{
                              color: 'var(--success)',
                              fontSize: '0.75rem',
                              marginLeft: 'auto',
                            }}
                          >
                            ✓
                          </span>
                        )}
                      </div>
                    ))}
                  </div>

                  <Progress value={evaluationProgress} />

                  <p
                    className="text-small"
                    style={{
                      color: 'var(--text-muted)',
                      textAlign: 'center',
                    }}
                  >
                    Processing... Step {evaluationStep + 1} of {PIPELINE_STAGES.length}
                  </p>
                </div>
              )}

              {/* Out of context rejection */}
              {apiEvaluationResult?.status === 'out_of_context' && !isEvaluating && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-md)',
                    padding: 'var(--space-lg) 0',
                  }}
                >
                  <div
                    style={{
                      width: 80,
                      height: 80,
                      borderRadius: '50%',
                      border: '4px solid #ef4444',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 0 20px #ef444440',
                    }}
                  >
                    <span style={{ fontSize: '2rem' }}>0</span>
                  </div>
                  <p
                    className="heading-4"
                    style={{ textAlign: 'center', color: '#ef4444' }}
                  >
                    Out of Context
                  </p>
                  <p
                    className="text-small"
                    style={{
                      textAlign: 'center',
                      color: 'var(--text-secondary)',
                      maxWidth: 280,
                    }}
                  >
                    {apiEvaluationResult.message}
                  </p>
                  <button
                    className="btn btn-secondary"
                    style={{ width: '100%' }}
                    onClick={() => {
                      resetEvaluation();
                      setApiEvaluationResult(null);
                    }}
                  >
                    <RotateCcw size={14} />
                    Try Again
                  </button>
                </div>
              )}

              {/* Complete */}
              {apiEvaluationResult && apiEvaluationResult.status !== 'out_of_context' && !isEvaluating && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-md)',
                    padding: 'var(--space-md) 0',
                  }}
                >
                  {/* Score circle */}
                  <div className="flex-center">
                    <div
                      style={{
                        width: 100,
                        height: 100,
                        borderRadius: '50%',
                        border: `4px solid ${scoreColor(apiEvaluationResult.score)}`,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: `0 0 20px ${scoreColor(apiEvaluationResult.score)}40`,
                      }}
                    >
                      <span
                        style={{
                          fontSize: '2rem',
                          fontWeight: 700,
                          color: scoreColor(apiEvaluationResult.score),
                        }}
                      >
                        {apiEvaluationResult.score}
                      </span>
                      <span className="label">{scoreLabel(apiEvaluationResult.score)}</span>
                    </div>
                  </div>

                  {/* Metric cards */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: 'var(--space-sm)',
                    }}
                  >
                    {(
                      [
                        {
                          label: 'Angle Dev.',
                          value: apiEvaluationResult.metrics?.angle_deviation,
                          unit: '',
                        },
                        {
                          label: 'Trajectory',
                          value: apiEvaluationResult.metrics?.trajectory_deviation,
                          unit: '',
                        },
                        {
                          label: 'Velocity',
                          value: apiEvaluationResult.metrics?.velocity_difference,
                          unit: '',
                        },
                        {
                          label: 'Tool Align.',
                          value: apiEvaluationResult.metrics?.tool_alignment_deviation,
                          unit: '',
                        },
                      ] as const
                    ).map((m) => (
                      <div
                        key={m.label}
                        className="stat-card"
                        style={{ padding: 'var(--space-sm)' }}
                      >
                        <div
                          className="stat-value"
                          style={{ fontSize: '1.25rem' }}
                        >
                          {typeof m.value === 'number' ? m.value : 0}
                          {m.unit}
                        </div>
                        <div
                          className="stat-label"
                          style={{ fontSize: '0.7rem' }}
                        >
                          {m.label}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* AI explanation */}
                  <div
                    style={{
                      background: 'var(--bg-tertiary)',
                      borderRadius: 'var(--radius-md)',
                      padding: 'var(--space-md)',
                    }}
                  >
                    <p
                      className="text-small"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {apiEvaluationResult.explanation?.explanation ?? ''}
                    </p>
                    {Array.isArray(apiEvaluationResult.key_error_moments) &&
                      apiEvaluationResult.key_error_moments.length > 0 && (
                        <div style={{ marginTop: 'var(--space-sm)' }}>
                          {apiEvaluationResult.key_error_moments.map((error: any, index: number) => (
                            <p
                              key={`${error.error_type ?? 'error'}-${index}`}
                              className="text-small"
                              style={{ color: 'var(--text-secondary)' }}
                            >
                              - {error.semantic_label || error.label}
                            </p>
                          ))}
                        </div>
                      )}
                  </div>

                  <button
                    className="btn btn-secondary"
                    style={{ width: '100%' }}
                    onClick={() => {
                      resetEvaluation();
                      setApiEvaluationResult(null);
                    }}
                  >
                    <RotateCcw size={14} />
                    Run Again
                  </button>
                </div>
              )}
            </TabsContent>

            {/* ── Tab: Timers ──────────────────────────────────────────── */}
            <TabsContent value="timers">
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 'var(--space-md)',
                }}
              >
                {timers.map((timer) => (
                  <div
                    key={timer.id}
                    className="card"
                    style={{ padding: 'var(--space-md)' }}
                  >
                    <div
                      className="flex-between"
                      style={{ marginBottom: 'var(--space-sm)' }}
                    >
                      <span
                        className="text-small"
                        style={{ fontWeight: 600 }}
                      >
                        {timer.name}
                      </span>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '1.25rem',
                          fontWeight: 700,
                          color: timer.isRunning
                            ? 'var(--accent-primary)'
                            : 'var(--text-primary)',
                        }}
                      >
                        {formatTime(timer.elapsed)}
                      </span>
                    </div>

                    <div
                      style={{
                        display: 'flex',
                        gap: 'var(--space-xs)',
                      }}
                    >
                      <button
                        className={`btn ${timer.isRunning ? 'btn-secondary' : 'btn-primary'}`}
                        style={{ flex: 1, fontSize: '0.75rem' }}
                        onClick={() =>
                          timer.isRunning
                            ? stopTimer(timer.id)
                            : startTimer(timer.id)
                        }
                      >
                        {timer.isRunning ? (
                          <Pause size={12} />
                        ) : (
                          <Play size={12} />
                        )}
                        {timer.isRunning ? 'Stop' : 'Start'}
                      </button>
                      <button
                        className="btn btn-ghost"
                        style={{ fontSize: '0.75rem' }}
                        onClick={() => resetTimer(timer.id)}
                      >
                        <RotateCcw size={12} />
                      </button>
                      {timer.isRunning && (
                        <button
                          className="btn btn-ghost"
                          style={{ fontSize: '0.75rem' }}
                          onClick={() =>
                            addTimestamp(timer.id, timer.elapsed)
                          }
                        >
                          <Flag size={12} />
                        </button>
                      )}
                    </div>

                    {timer.timestamps.length > 0 && (
                      <div
                        style={{
                          marginTop: 'var(--space-sm)',
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: 'var(--space-xs)',
                        }}
                      >
                        {timer.timestamps.map((ts, i) => (
                          <span
                            key={i}
                            className="badge badge-blue"
                            style={{
                              fontFamily: 'var(--font-mono)',
                              fontSize: '0.65rem',
                            }}
                          >
                            #{i + 1} {formatTime(ts)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}

                <button
                  className="btn btn-secondary"
                  style={{ width: '100%' }}
                  onClick={handleAddTimer}
                >
                  <Plus size={14} />
                  Add Timer
                </button>
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>

      {/* ─── Guided Tour Overlay ───────────────────────────────────────── */}
      <AnimatePresence>
        {showGuidedTour && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="modal-content"
              style={{
                padding: 'var(--space-2xl)',
                maxWidth: 440,
                textAlign: 'center',
              }}
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              key={tourStep}
            >
              <h3
                className="heading-3"
                style={{ marginBottom: 'var(--space-sm)' }}
              >
                {TOUR_STEPS[tourStep]?.title}
              </h3>
              <p
                className="text-body"
                style={{ marginBottom: 'var(--space-lg)' }}
              >
                {TOUR_STEPS[tourStep]?.description}
              </p>

              {/* Step dots */}
              <div
                className="flex-center"
                style={{
                  gap: 'var(--space-sm)',
                  marginBottom: 'var(--space-lg)',
                }}
              >
                {TOUR_STEPS.map((_, i) => (
                  <div
                    key={i}
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      background:
                        i === tourStep
                          ? 'var(--accent-primary)'
                          : 'var(--bg-tertiary)',
                      transition: 'background var(--transition-fast)',
                    }}
                  />
                ))}
              </div>

              {/* Navigation */}
              <div
                style={{
                  display: 'flex',
                  gap: 'var(--space-sm)',
                  justifyContent: 'center',
                }}
              >
                {tourStep > 0 && (
                  <button className="btn btn-ghost" onClick={prevTourStep}>
                    Back
                  </button>
                )}
                {tourStep < TOUR_STEPS.length - 1 ? (
                  <button className="btn btn-primary" onClick={nextTourStep}>
                    Next
                  </button>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={() => setShowGuidedTour(false)}
                  >
                    Get Started
                  </button>
                )}
                <button
                  className="btn btn-ghost"
                  onClick={() => setShowGuidedTour(false)}
                >
                  Skip
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
