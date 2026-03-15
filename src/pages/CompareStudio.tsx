import { useState, useRef, useEffect, useCallback } from 'react';
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
import {
  generateMockEvaluation,
  scoreColor,
} from '../services/mock/evaluation';
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

  const { selectedCourse, selectedClip, userVideo, setUserVideo } =
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
    evaluationResult,
    startEvaluation,
    setEvaluationStep,
    setEvaluationProgress,
    setEvaluationResult,
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
  const [expertCurrentTime, setExpertCurrentTime] = useState(0);
  const [expertDuration, setExpertDuration] = useState(0);
  const [learnerCurrentTime, setLearnerCurrentTime] = useState(0);
  const [learnerDuration, setLearnerDuration] = useState(0);
  const [expertMuted, setExpertMuted] = useState(false);
  const [learnerMuted, setLearnerMuted] = useState(false);

  // ── Refs ──────────────────────────────────────────────────────────────────

  const expertVideoRef = useRef<HTMLVideoElement>(null);
  const learnerVideoRef = useRef<HTMLVideoElement>(null);
  const expertCanvasRef = useRef<HTMLCanvasElement>(null);
  const timersRef = useRef(timers);
  timersRef.current = timers;

  // ── Derived data ─────────────────────────────────────────────────────────

  const course = courses.find((c) => c.id === selectedCourse);
  const clips = selectedCourse ? getClipsForCourse(selectedCourse) : [];
  const clip = clips.find((c) => c.id === selectedClip);

  // ── Effects ──────────────────────────────────────────────────────────────

  useEffect(() => {
    setRobotMessage(
      clip
        ? `Practicing "${clip.title}". Upload your video and compare with the expert!`
        : 'Select a clip from a course to start comparing. Head to Courses to pick one!',
    );
    return () => setRobotMessage(null);
  }, [clip, setRobotMessage]);

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

  // Revoke objectURL on unmount
  useEffect(() => {
    return () => {
      if (userVideoUrl) URL.revokeObjectURL(userVideoUrl);
    };
  }, [userVideoUrl]);

  // ── Video upload ─────────────────────────────────────────────────────────

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      const validTypes = ['video/mp4', 'video/quicktime'];
      if (!validTypes.includes(file.type)) {
        toast.error('Please upload an MP4 or MOV file');
        return;
      }

      const url = URL.createObjectURL(file);
      const probe = document.createElement('video');
      probe.preload = 'metadata';
      probe.onloadedmetadata = () => {
        if (probe.duration > 120) {
          toast.error('Video must be 2 minutes or shorter');
          URL.revokeObjectURL(url);
          return;
        }
        setUserVideoUrl(url);
        setUserVideo(file);
        toast.success('Video uploaded successfully!');
      };
      probe.src = url;
    },
    [setUserVideo],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/mp4': ['.mp4'], 'video/quicktime': ['.mov'] },
    maxFiles: 1,
  });

  // ── Evaluation pipeline ──────────────────────────────────────────────────

  const runEvaluation = useCallback(async () => {
    if (!selectedCourse || !selectedClip || !userVideo) return;
    startEvaluation();

    const total = PIPELINE_STAGES.length;
    const sub = 10;
    for (let i = 0; i < total; i++) {
      setEvaluationStep(i);
      for (let p = 0; p <= sub; p++) {
        setEvaluationProgress(((i * sub + p) / (total * sub)) * 100);
        await new Promise((r) => setTimeout(r, 120 + Math.random() * 200));
      }
    }

    const result = generateMockEvaluation(selectedCourse, selectedClip);
    setEvaluationResult(result);
    toast.success(`Evaluation complete! Score: ${result.score}/100`);
  }, [
    selectedCourse,
    selectedClip,
    userVideo,
    startEvaluation,
    setEvaluationStep,
    setEvaluationProgress,
    setEvaluationResult,
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

          {clip ? (
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
                  poster={clip.thumbnail}
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
                <p className="empty-state-title">No Clip Selected</p>
                <p className="empty-state-description">
                  Select a clip from a course to begin
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
          <span
            className="label"
            style={{ display: 'block', marginBottom: 'var(--space-sm)' }}
          >
            Your Practice
          </span>

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
              {!isEvaluating && !evaluationResult && (
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
                    Step {evaluationStep + 1} of {PIPELINE_STAGES.length}
                  </p>
                </div>
              )}

              {/* Complete */}
              {evaluationResult && !isEvaluating && (
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
                        border: `4px solid ${scoreColor(evaluationResult.score)}`,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: `0 0 20px ${scoreColor(evaluationResult.score)}40`,
                      }}
                    >
                      <span
                        style={{
                          fontSize: '2rem',
                          fontWeight: 700,
                          color: scoreColor(evaluationResult.score),
                        }}
                      >
                        {evaluationResult.score}
                      </span>
                      <span className="label">{evaluationResult.label}</span>
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
                          value: evaluationResult.metrics.angleDeviation,
                          unit: '°',
                        },
                        {
                          label: 'Trajectory',
                          value: evaluationResult.metrics.trajectoryDeviation,
                          unit: '°',
                        },
                        {
                          label: 'Velocity',
                          value: evaluationResult.metrics.velocityDifference,
                          unit: '%',
                        },
                        {
                          label: 'Tool Align.',
                          value: evaluationResult.metrics.toolAlignmentDeviation,
                          unit: '°',
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
                          {m.value}
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
                      {evaluationResult.explanation}
                    </p>
                  </div>

                  <button
                    className="btn btn-secondary"
                    style={{ width: '100%' }}
                    onClick={resetEvaluation}
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
