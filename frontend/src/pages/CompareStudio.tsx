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
  Activity,
  Loader2,
  Hand,
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
import { fetchClipsForCourse, fetchCourse } from '../services/api/courses';
import { getEvaluationResult, startEvaluation as startEvaluationApi } from '../api/evaluationApi';
import { useAuth } from '@/contexts/AuthContext';
import { runLearnerAngle, generateDtwPreview } from '../api/angleApi';
import type { AngleDtwSummary } from '../api/angleApi';
import { formatTime } from '../utils/helpers';
import type { DrawingTool, VideoClip } from '../types';

// ── Constants ────────────────────────────────────────────────────────────────

const PIPELINE_STAGES = [
  'Validation',
  'Pose Analysis',
  'DTW Alignment',
  'Metric Computation',
  'VLM Explanation',
];

const EVAL_STEPS = [
  'Uploading your video',
  'Detecting scissors in your video',
  'Tracking scissor path',
  'Detecting trajectory errors',
  'Analyzing cutting angles',
  'Comparing angles to expert',
  'Analyzing hand vibration',
  'Analysis complete',
];

const EVAL_HINTS = [
  'Analyzing your cutting path frame by frame...',
  'Comparing your trajectory to the expert...',
  'Measuring your cutting angles...',
  'Almost there...',
];

const SSE_STEP_INDEX: Record<string, number> = {
  yolo: 1,
  trajectory_init: 2,
  trajectory_track: 2,
  trajectory_errors: 3,
  angle_init: 4,
  angle_track: 5,
  angle_errors: 5,
  vibration: 6,
  done: 7,
};

const SSE_PROGRESS: Record<string, number> = {
  yolo: 20,
  trajectory_init: 35,
  trajectory_track: 50,
  trajectory_errors: 60,
  angle_init: 72,
  angle_track: 86,
  angle_errors: 88,
  vibration: 95,
  done: 100,
};


const PLAYBACK_RATES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

const toPlayableStorageUrl = (url?: string | null, path?: string | null) => {
  const candidate = url || path;
  if (!candidate) return null;
  const normalized = candidate.replace(/\\/g, '/');
  if (normalized.startsWith('http://') || normalized.startsWith('https://') || normalized.startsWith('/storage/')) {
    return normalized;
  }
  if (normalized.startsWith('storage/')) {
    return `/${normalized}`;
  }
  const storageIndex = normalized.toLowerCase().lastIndexOf('/storage/');
  if (storageIndex >= 0) {
    return normalized.slice(storageIndex);
  }
  return normalized;
};

// ── MediaPipe integration ──────────────────────────────────────────────────

interface MediaPipeRunSummary {
  run_id: string;
  source_video_path?: string;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  created_at: string;
  selected_hand_policy: string;
  total_frames: number;
  frames_with_detection: number;
  detection_rate: number;
  right_hand_selected_count: number;
  left_hand_selected_count: number;
}

interface MediaPipeRunResult {
  run_id: string;
  run_folder: string;
  detections_json_path: string;
  features_json_path: string;
  metadata_json_path: string;
  annotated_video_path: string;
  annotated_video_url: string | null;
  summary: MediaPipeRunSummary;
  partial_errors?: string[];
}

type InspectionModel = 'mediapipe' | 'sam2' | 'optical_flow' | 'yolo_angle';

const DEFAULT_SAM2_YOLO_EXPERT_CODE = 'straight_line_v1';

interface Sam2LearnerMediaPipeInfo {
  run_id: string;
  run_folder: string;
  features_json_path: string;
  metadata_json_path: string;
  annotated_video_path: string;
  annotated_video_url: string | null;
  total_frames: number;
  frames_with_detection: number;
  detection_rate: number;
}

interface Sam2LearnerMetadata {
  run_id: string;
  source_video_path: string;
  pipeline_name: string;
  model_name: string;
  device: string;
  gpu_name?: string | null;
  model_checkpoint_path?: string | null;
  model_config_path?: string | null;
  created_at: string;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  analysis_start_frame_index: number;
  analysis_end_frame_index: number;
  frame_stride: number;
  total_frames_processed: number;
  frames_with_mask: number;
  detection_rate: number;
  target_object_id: number;
  init_prompt: unknown;
  warnings: string[];
}

interface Sam2LearnerSummary {
  run_id: string;
  pipeline_name: string;
  model_name: string;
  total_frames: number;
  frames_with_mask: number;
  detection_rate: number;
  mean_mask_area_px: number | null;
  min_mask_area_px: number | null;
  max_mask_area_px: number | null;
  std_mask_area_px: number | null;
  mean_centroid_speed_px_per_frame: number | null;
  track_fragmentation_count: number;
  working_region: unknown;
  warnings: string[];
}

interface Sam2LearnerResult {
  run_id: string;
  run_folder?: string;
  run_dir?: string;
  status?: string;
  failure_reason?: string;
  model?: string;
  raw_json_path: string;
  raw_json_url?: string | null;
  metrics_json_path?: string;
  metrics_json_url?: string | null;
  summary_json_path: string;
  summary_json_url?: string | null;
  metadata_json_path?: string;
  annotated_video_path?: string | null;
  annotated_video_url?: string | null;
  overlay_video_path?: string | null;
  overlay_video_url?: string | null;
  source_video_path: string;
  video_path?: string;
  initialization_debug_image_path?: string | null;
  initialization_debug_image_url?: string | null;
  tip_initialization_json_path?: string | null;
  tip_initialization_json_url?: string | null;
  tip_tracking_json_path?: string | null;
  tip_tracking_json_url?: string | null;
  tip_annotated_video_path?: string | null;
  tip_annotated_video_url?: string | null;
  manual_init_prompt_path?: string | null;
  manual_init_prompt_url?: string | null;
  device: string;
  frame_stride: number;
  original_frame_count?: number | null;
  expected_strided_frames?: number | null;
  max_processed_frames?: number | null;
  processed_frames?: number;
  successful_masks?: number;
  failure_frames?: number[];
  prompt_source?: string;
  main_tracked_feature?: string;
  trajectory_stability_score?: number;
  horizontal_drift_range?: number;
  region_stability_score?: number;
  tracking_quality_note?: string;
  expert_reference_available?: boolean;
  expert_code?: string | null;
  expert_warning?: string;
  expert_raw_json_path?: string | null;
  expert_metrics_json_path?: string | null;
  metadata?: Sam2LearnerMetadata;
  summary?: Sam2LearnerSummary;
  trajectory_metrics?: unknown;
  region_metrics?: unknown;
  quality_flags?: unknown;
  raw_preview?: unknown | null;
  tip_tracking_preview?: unknown | null;
  manual_init_prompt_preview?: unknown | null;
  mediapipe?: Sam2LearnerMediaPipeInfo | null;
  warnings: string[];
  // Aligned corridor (Step 4 post-processing)
  aligned_corridor_progress_json_path?: string | null;
  aligned_corridor_progress_json_url?: string | null;
  aligned_corridor_progress_preview_path?: string | null;
  aligned_corridor_progress_preview_url?: string | null;
  aligned_corridor_overlay_video_path?: string | null;
  aligned_corridor_overlay_video_url?: string | null;
  alignment_mode?: string | null;
  alignment_expert_code?: string | null;
  aligned_corridor_error?: string | null;
  aligned_corridor_warning?: string | null;
}

interface OpticalFlowSummaryMetrics {
  avg_magnitude: number;
  motion_stability_score: number;
  vibration_score: number;
  vibration_high_freq_mean: number;
  magnitude_jitter: number;
  roi_usage_ratio: number;
}

interface OpticalFlowLearnerResult {
  run_id: string;
  learner_video_path: string;
  learner_video_url?: string | null;
  raw_json_path: string;
  raw_json_url?: string | null;
  summary_json_path: string;
  summary_json_url?: string | null;
  visualization_video_path: string | null;
  visualization_video_url?: string | null;
  summary: OpticalFlowSummaryMetrics;
}

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

function prettyJson(value: unknown): string {
  if (value === null || value === undefined) return '(no data)';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function storagePathToUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (path.startsWith('/storage/')) return path;
  const normalized = path.replace(/\\/g, '/');
  const marker = '/storage/';
  const markerIndex = normalized.lastIndexOf(marker);
  if (markerIndex >= 0) {
    return normalized.slice(markerIndex);
  }
  return normalized;
}

function formatMetricValue(value: unknown): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function renderFeedback(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/).map((chunk, i) =>
    chunk.startsWith('**') && chunk.endsWith('**')
      ? <strong key={i}>{chunk.slice(2, -2)}</strong>
      : chunk,
  );
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

  const { user } = useAuth();

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

  // ── Evaluate tab — new progress-UI state ─────────────────────────────────
  type EvalPhase = 'idle' | 'uploading' | 'streaming' | 'done' | 'error';
  const [evalPhase, setEvalPhase] = useState<EvalPhase>('idle');
  const [evalStepIndex, setEvalStepIndex] = useState(0);
  const [evalProgress, setEvalProgress] = useState(0);
  const [evalHintIndex, setEvalHintIndex] = useState(0);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalRunId, setEvalRunId] = useState<string | null>(null);
  const [evalEvaluationId, setEvalEvaluationId] = useState<string | null>(null);
  const evalEventSourceRef = useRef<EventSource | null>(null);

  // ── Game phase state ───────────────────────────────────────────────────────
  type GamePhase = 'idle' | 'processing' | 'game' | 'result';
  type MarkType = 'trajectory' | 'angle' | 'vibration';
  interface UserMark {
    id: string;
    mark_type: MarkType;
    x: number;
    y: number;
    display_x: number;
    display_y: number;
    timestamp_sec: number;
    video_width: number;
    video_height: number;
  }
  interface ScoreResult {
    run_id: string;
    total_real_errors: number;
    correct_marks: number;
    false_alarms: number;
    missed_errors: number;
    score_pct: number;
    mark_results: Array<{
      mark_type: string;
      x: number;
      y: number;
      timestamp_sec: number | null;
      result: 'correct' | 'false_alarm';
      matched_error_id: number | null;
    }>;
    missed_error_details: Array<{
      error_id: number;
      error_type: string;
      timestamp_start_sec: number;
      timestamp_end_sec: number;
      peak_location: { x: number; y: number };
    }>;
  }
  interface UnifiedError {
    error_id: number;
    error_type: 'trajectory' | 'angle' | 'vibration';
    timestamp_start_sec: number;
    timestamp_end_sec: number;
    duration_sec: number;
    peak_location: { x: number; y: number } | null;
    bounding_box: { x_min: number; y_min: number; x_max: number; y_max: number } | null;
    dominant_freq_hz?: number | null;
    peak_confidence?: number | null;
    severity?: 'mild' | 'moderate' | 'severe' | null;
    consecutive_windows?: number | null;
  }

  const [gamePhase, setGamePhase] = useState<GamePhase>('idle');
  const [activeMarkType, setActiveMarkType] = useState<MarkType | null>(null);
  const [userMarks, setUserMarks] = useState<UserMark[]>([]);
  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [gameErrors, setGameErrors] = useState<UnifiedError[]>([]);
  const [feedbackText, setFeedbackText] = useState<string | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'loading' | 'done'>('idle');
  const feedbackPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // MediaPipe integration state.
  const [mediapipeRun, setMediapipeRun] = useState<MediaPipeRunResult | null>(null);
  const [isMediapipeProcessing, setIsMediapipeProcessing] = useState(false);
  const [mediapipeError, setMediapipeError] = useState<string | null>(null);
  const [mediapipeVideoVersion, setMediapipeVideoVersion] = useState(0);
  const [selectedExpertVideoId, setSelectedExpertVideoId] = useState<string | null>(null);
  const [inspectionModel, setInspectionModel] = useState<InspectionModel>('mediapipe');
  // Learner overlay: show the raw uploaded video, the MediaPipe annotated
  // output, YOLO+SAM2 scissors overlay, or Optical Flow visualization.
  const [learnerOverlay, setLearnerOverlay] =
    useState<'none' | 'mediapipe' | 'sam2' | 'optical_flow' | 'aligned_corridor' | 'angle' | 'eval_corridor' | 'visualization'>('none');

  // Path Overlay (on-demand corridor overlay from evaluation run).
  type PathOverlayState = 'disabled' | 'idle' | 'loading' | 'ready';
  const [pathOverlayState, setPathOverlayState] = useState<PathOverlayState>('disabled');
  const [evalCorridorOverlayUrl, setEvalCorridorOverlayUrl] = useState<string | null>(null);

  // Visualization (corridor lines + ghost scissor overlay).
  type VizState = 'idle' | 'loading' | 'ready';
  const [vizState, setVizState] = useState<VizState>('idle');
  const [vizUrl, setVizUrl] = useState<string | null>(null);

  // YOLO+SAM2 learner scissors tracking state.
  const [sam2LearnerRun, setSam2LearnerRun] = useState<Sam2LearnerResult | null>(null);
  const [isSam2LearnerProcessing, setIsSam2LearnerProcessing] = useState(false);
  const [sam2LearnerError, setSam2LearnerError] = useState<string | null>(null);
  const [sam2OverlayVideoError, setSam2OverlayVideoError] = useState<string | null>(null);
  const [sam2LearnerVideoVersion, setSam2LearnerVideoVersion] = useState(0);
  const [showSam2RawPreview, setShowSam2RawPreview] = useState(false);
  const [isTipTracking, setIsTipTracking] = useState(false);
  const [tipTrackingError, setTipTrackingError] = useState<string | null>(null);
  const [isSelectingTip, setIsSelectingTip] = useState(false);
  const [isManualSamInitMode, setIsManualSamInitMode] = useState(false);
  const [manualSamPointMode, setManualSamPointMode] = useState<'positive' | 'negative'>('positive');
  const [manualSamPrompt, setManualSamPrompt] = useState<{
    frame_index: number;
    box: [number, number, number, number] | null;
    positive_points: [number, number][];
    negative_points: [number, number][];
  }>({
    frame_index: 0,
    box: null,
    positive_points: [],
    negative_points: [],
  });
  const [manualSamDraftBox, setManualSamDraftBox] = useState<[number, number, number, number] | null>(null);
  const manualSamDragStartRef = useRef<[number, number] | null>(null);

  // Optical Flow learner-only analysis state.
  const [opticalFlowRun, setOpticalFlowRun] = useState<OpticalFlowLearnerResult | null>(null);
  const [isOpticalFlowProcessing, setIsOpticalFlowProcessing] = useState(false);
  const [opticalFlowError, setOpticalFlowError] = useState<string | null>(null);
  const [opticalFlowVideoVersion, setOpticalFlowVideoVersion] = useState(0);
  const [clips, setClips] = useState<VideoClip[]>([]);

  // YOLO+Angle+DTW state.
  const [isAngleProcessing, setIsAngleProcessing] = useState(false);
  const [angleStatus, setAngleStatus] = useState<'idle' | 'extracting' | 'dtw' | 'done'>('idle');
  const [angleDtwResult, setAngleDtwResult] = useState<{
    normalized_dtw_distance: number | null;
    mean_angle_difference: number | null;
    high_error_frame_count: number | null;
    medium_error_frame_count: number | null;
    ok_frame_count: number | null;
  } | null>(null);
  const [angleError, setAngleError] = useState<string | null>(null);
  // run_id returned by the backend for the most recent angle pipeline run.
  const [angleRunId, setAngleRunId] = useState<string | null>(null);

  // SYNC dialog + DTW preview state.
  const [isSyncDialogOpen, setIsSyncDialogOpen] = useState(false);
  const [isDtwPreviewGenerating, setIsDtwPreviewGenerating] = useState(false);
  const [dtwPreviewUrl, setDtwPreviewUrl] = useState<string | null>(null);

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

  const requestedCourseId = searchParams.get('courseId');
  const requestedClipId = searchParams.get('clipId');
  const clip = clips.find((c) => c.id === selectedClip);
  const hasSelectedExpertReference = Boolean(selectedClip || requestedClipId);

  // ── Effects ──────────────────────────────────────────────────────────────

  // Rotate hint message while evaluation is streaming.
  useEffect(() => {
    if (evalPhase !== 'streaming') return;
    const id = setInterval(() => {
      setEvalHintIndex((prev) => (prev + 1) % EVAL_HINTS.length);
    }, 4000);
    return () => clearInterval(id);
  }, [evalPhase]);

  // Clean up any open SSE connection on unmount.
  useEffect(() => {
    return () => {
      evalEventSourceRef.current?.close();
    };
  }, []);

  // Restore full CompareStudio state on mount (survives navigation, cleared on browser refresh).
  useEffect(() => {
    const saved = sessionStorage.getItem('augmentor_compare_state');
    if (saved) {
      try {
        const s = JSON.parse(saved) as {
          evalPhase?: string;
          evalRunId?: string | null;
          evalEvaluationId?: string | null;
          gamePhase?: string;
          scoreResult?: ScoreResult | null;
          userMarks?: UserMark[];
          gameErrors?: UnifiedError[];
          practiceVideoUrl?: string | null;
        };
        if (s.evalPhase && s.evalPhase !== 'idle') setEvalPhase(s.evalPhase as EvalPhase);
        if (s.evalRunId) setEvalRunId(s.evalRunId);
        if (s.evalEvaluationId) setEvalEvaluationId(s.evalEvaluationId);
        if (s.gamePhase && s.gamePhase !== 'idle') setGamePhase(s.gamePhase as GamePhase);
        if (s.scoreResult) setScoreResult(s.scoreResult);
        if (s.userMarks?.length) setUserMarks(s.userMarks);
        if (s.gameErrors?.length) setGameErrors(s.gameErrors);
        if (s.practiceVideoUrl) setUserVideoUrl(s.practiceVideoUrl);

        // If the eval was still streaming when the user navigated away, re-attach the SSE.
        if (s.evalPhase === 'streaming' && s.evalEvaluationId) {
          const evaluationId = s.evalEvaluationId;
          const es = new EventSource(`/api/evaluations/${evaluationId}/status-stream`);
          evalEventSourceRef.current = es;

          es.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data as string) as {
                step?: string;
                progress?: number;
                run_id?: string;
              };
              if (data.run_id) setEvalRunId(data.run_id);
              const step = data.step ?? '';
              const progress = data.progress ?? SSE_PROGRESS[step] ?? undefined;
              if (step in SSE_STEP_INDEX) setEvalStepIndex(SSE_STEP_INDEX[step]);
              if (progress !== undefined) setEvalProgress(progress);
              if (step === 'done' || progress === 100) {
                es.close();
                evalEventSourceRef.current = null;
                setEvalStepIndex(7);
                setEvalProgress(100);
                setEvalPhase('done');
                setGamePhase('game');
              }
            } catch {
              // ignore malformed events
            }
          };

          es.onerror = () => {
            es.close();
            evalEventSourceRef.current = null;
            setEvalPhase('error');
            setEvalError('Connection lost. Please try again.');
          };
        }
      } catch {
        sessionStorage.removeItem('augmentor_compare_state');
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist key state to sessionStorage so it survives SPA navigation (cleared on browser refresh).
  useEffect(() => {
    sessionStorage.setItem('augmentor_compare_state', JSON.stringify({
      evalPhase,
      evalRunId,
      evalEvaluationId,
      gamePhase,
      scoreResult,
      userMarks,
      gameErrors,
      practiceVideoUrl: userVideoUrl,
    }));
  }, [evalPhase, evalRunId, evalEvaluationId, gamePhase, scoreResult, userMarks, gameErrors, userVideoUrl]);

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
    let isCancelled = false;

    const loadSelectedCourse = async () => {
      if (!selectedCourse) {
        setClips([]);
        return;
      }

      try {
        const backendCourse = await fetchCourse(selectedCourse);
        const backendClips = await fetchClipsForCourse(selectedCourse, backendCourse.thumbnail);
        if (!isCancelled) {
          setClips(backendClips);
        }
      } catch {
        if (!isCancelled) {
          setClips([]);
        }
      }
    };

    void loadSelectedCourse();

    return () => {
      isCancelled = true;
    };
  }, [selectedCourse]);

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
      setSelectedExpertVideoId(null);
      setExpertCurrentTime(0);
      setExpertDuration(0);
    }
  }, [clip]);

  useEffect(() => {
    let isCancelled = false;

    const loadSelectedExpertVideo = async () => {
      try {
        // 1) If selectedClip is a real backend chapter id, resolve chapter expert video first.
        // This ensures Compare Studio uses the latest backend expert reference,
        // not a stale static mock path.
        if (selectedClip) {
          const chapterResponse = await fetch(
            `/api/chapters/${encodeURIComponent(selectedClip)}/expert-video`,
          );
          if (chapterResponse.ok) {
            const chapterPayload = (await chapterResponse.json()) as {
              id?: string;
              url: string;
            };
            if (!isCancelled) {
              setExpertVideoError(null);
              setExpertVideoUrl(chapterPayload.url);
              setSelectedExpertVideoId(chapterPayload.id ?? null);
            }
            return;
          }
        }

        // 2) Fallback to backend default expert video (still backend-managed).
        const defaultResponse = await fetch('/api/chapters/default/expert-video');
        if (defaultResponse.ok) {
          const defaultPayload = (await defaultResponse.json()) as {
            expert_video_id?: string;
            url: string;
          };
          if (!isCancelled) {
            setExpertVideoError(null);
            setExpertVideoUrl(defaultPayload.url);
            setSelectedExpertVideoId(defaultPayload.expert_video_id ?? null);
          }
          return;
        }

        // 3) Fallback to the expert URL already included in the backend chapter clip.
        if (clip?.expertVideoUrl) {
          if (!isCancelled) {
            setExpertVideoError(null);
            setExpertVideoUrl(clip.expertVideoUrl);
            setSelectedExpertVideoId(null);
          }
          return;
        }

        if (!isCancelled) {
          setExpertVideoUrl(null);
          setSelectedExpertVideoId(null);
          setExpertVideoError('No expert video is linked yet for the selected chapter.');
        }
      } catch {
        if (!isCancelled) {
          setExpertVideoUrl(null);
          setSelectedExpertVideoId(null);
          setExpertVideoError('No expert video is linked yet for the selected chapter.');
        }
      }
    };

    void loadSelectedExpertVideo();

    return () => {
      isCancelled = true;
    };
  }, [clip, selectedClip]);

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

  // Load unified errors (for vibration bbox overlay) once the game phase starts.
  useEffect(() => {
    if (gamePhase !== 'game' || !evalEvaluationId || !evalRunId) return;
    let cancelled = false;
    fetch(`/api/evaluations/${evalEvaluationId}/errors?run_id=${evalRunId}`)
      .then((r) => r.json())
      .then((d: { all_errors?: UnifiedError[] }) => {
        if (!cancelled) setGameErrors(d.all_errors ?? []);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [gamePhase, evalEvaluationId, evalRunId]);

  // Poll for VLM coaching feedback once the score result panel appears.
  useEffect(() => {
    if (gamePhase !== 'result' || !evalEvaluationId || !evalRunId || !selectedClip) return;

    setFeedbackStatus('loading');
    setFeedbackText(null);
    let cancelled = false;
    let attempts = 0;

    const poll = async () => {
      if (cancelled) return;
      attempts++;
      if (attempts > 60) {
        clearInterval(pollId);
        setFeedbackStatus('done');
        setFeedbackText('Your Crafting Coach is taking longer than expected. Please try again later.');
        return;
      }
      try {
        const res = await fetch(
          `/api/evaluations/${encodeURIComponent(evalEvaluationId)}/generate-feedback` +
          `?run_id=${encodeURIComponent(evalRunId)}&expert_id=${encodeURIComponent(selectedClip)}`,
        );
        if (cancelled || !res.ok) return;
        const data = await res.json() as { status: string; feedback?: string };
        if (!cancelled && data.status === 'done' && data.feedback) {
          setFeedbackText(data.feedback);
          setFeedbackStatus('done');
          cancelled = true;
          if (feedbackPollRef.current) {
            clearInterval(feedbackPollRef.current);
            feedbackPollRef.current = null;
          }
        }
      } catch { /* keep polling */ }
    };

    void poll();
    const pollId = setInterval(() => void poll(), 3000);
    feedbackPollRef.current = pollId;

    return () => {
      cancelled = true;
      clearInterval(pollId);
      feedbackPollRef.current = null;
    };
  }, [gamePhase, evalEvaluationId, evalRunId, selectedClip]);

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
        setMediapipeRun(null);
        setMediapipeError(null);
        setMediapipeVideoVersion(0);
        setSam2LearnerRun(null);
        setSam2LearnerError(null);
        setSam2LearnerVideoVersion(0);
        setTipTrackingError(null);
        setIsSelectingTip(false);
        setIsManualSamInitMode(false);
        setManualSamPrompt({
          frame_index: 0,
          box: null,
          positive_points: [],
          negative_points: [],
        });
        setManualSamDraftBox(null);
        setLearnerOverlay('none');
        setOpticalFlowRun(null);
        setOpticalFlowError(null);
        setOpticalFlowVideoVersion(0);
        sessionStorage.removeItem('augmentor_compare_state');
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

    // Close any leftover SSE connection.
    evalEventSourceRef.current?.close();
    evalEventSourceRef.current = null;

    setEvalPhase('uploading');
    setEvalStepIndex(0);
    setEvalProgress(8);
    setEvalHintIndex(0);
    setEvalError(null);
    setEvalRunId(null);
    setEvalEvaluationId(null);
    setApiEvaluationResult(null);
    setGamePhase('processing');
    setActiveMarkType(null);
    setUserMarks([]);
    setScoreResult(null);

    try {
      const formData = new FormData();
      formData.append('file', userVideo);
      formData.append('course_id', selectedCourse);
      formData.append('clip_id', selectedClip);
      formData.append('filename', userVideo.name);
      if (user?.id) formData.append('user_id', user.id);

      const started = await startEvaluationApi(formData);

      if (started?.status === 'out_of_context') {
        setApiEvaluationResult({
          score: 0,
          status: 'out_of_context',
          message: started.message,
          gate_reasons: started.gate_reasons || [],
          metrics: null,
          explanation: null,
          key_error_moments: [],
        });
        setEvalPhase('idle');
        toast.error('Video rejected: does not match the expert task');
        return;
      }

      const evaluationId =
        typeof started === 'string'
          ? started
          : started?.evaluation_id || started?.id;

      if (!evaluationId) throw new Error('Missing evaluation_id from backend response');

      setEvalEvaluationId(evaluationId);
      if (started?.run_id) setEvalRunId(started.run_id);

      // Persist the server-side video URL so the player can be restored after navigation.
      if (started?.video_url) {
        const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        setUserVideoUrl(`${API_BASE}${started.video_url}`);
      }

      // Upload done — step 0 complete, step 1 running.
      setEvalStepIndex(1);
      setEvalProgress(14);
      setEvalPhase('streaming');

      const es = new EventSource(`/api/evaluations/${evaluationId}/status-stream`);
      evalEventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string) as {
            step?: string;
            progress?: number;
            run_id?: string;
          };

          if (data.run_id) setEvalRunId(data.run_id);

          const step = data.step ?? '';
          const progress = data.progress ?? SSE_PROGRESS[step] ?? undefined;

          if (step in SSE_STEP_INDEX) {
            setEvalStepIndex(SSE_STEP_INDEX[step]);
          }
          if (progress !== undefined) {
            setEvalProgress(progress);
          }

          if (step === 'done' || progress === 100) {
            es.close();
            evalEventSourceRef.current = null;
            setEvalStepIndex(7);
            setEvalProgress(100);
            setEvalPhase('done');
            setPathOverlayState('idle');
            setGamePhase('game');
          }
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = () => {
        es.close();
        evalEventSourceRef.current = null;
        setEvalPhase('error');
        setEvalError('Connection lost. Please try again.');
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Evaluation failed';
      setEvalError(message);
      setEvalPhase('error');
    }
  }, [
    selectedCourse,
    selectedClip,
    userVideo,
  ]);

  // ── MediaPipe run ────────────────────────────────────────────────────────

  const runMediapipe = useCallback(async () => {
    if (!userVideo) {
      toast.error('Upload a practice video first');
      return;
    }

    setIsMediapipeProcessing(true);
    setMediapipeError(null);

    try {
      const formData = new FormData();
      formData.append('file', userVideo);
      formData.append('render_annotation', 'true');

      const response = await fetch('/api/mediapipe/process-upload', {
        method: 'POST',
        body: formData,
      });

      let payload: any = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail =
          payload?.detail ||
          payload?.message ||
          `MediaPipe failed with status ${response.status}`;
        throw new Error(detail);
      }

      const result = payload as MediaPipeRunResult;
      setMediapipeRun(result);
      setMediapipeVideoVersion(Date.now());
      if (result.annotated_video_url) {
        setLearnerOverlay('mediapipe');
      }

      const detectedPct = Math.round((result.summary.detection_rate || 0) * 100);
      toast.success(`MediaPipe ready — ${detectedPct}% frames detected`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'MediaPipe pipeline failed.';
      setMediapipeError(message);
      toast.error(message);
    } finally {
      setIsMediapipeProcessing(false);
    }
  }, [userVideo]);

  // ── YOLO+SAM2 scissors learner run ───────────────────────────────────────
  //
  // Triggered by the "Run YOLO+SAM2" button. This posts the uploaded learner
  // video directly to the YOLO+SAM2 backend; it does not call MediaPipe.
  const runSam2Learner = useCallback(async () => {
    if (!userVideo) {
      toast.error('Upload a practice video first');
      return;
    }

    setIsSam2LearnerProcessing(true);
    setSam2LearnerError(null);
    setSam2OverlayVideoError(null);
    setTipTrackingError(null);
    setIsSelectingTip(false);
    setIsManualSamInitMode(false);

    try {
      const formData = new FormData();
      formData.append('file', userVideo);
      formData.append('expert_code', DEFAULT_SAM2_YOLO_EXPERT_CODE);
      formData.append('stride', '5');
      formData.append('tracking_point_type', 'bbox_center');

      const response = await fetch('/api/sam2-yolo/run-learner', {
        method: 'POST',
        body: formData,
      });

      let payload: any = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail = payload?.detail;
        const message =
          (detail && typeof detail === 'object' && detail.message) ||
          (typeof detail === 'string' ? detail : null) ||
          payload?.failure_reason ||
          `YOLO+SAM2 failed with status ${response.status}`;
        throw new Error(message);
      }

      const result = payload as Sam2LearnerResult;
      setSam2LearnerRun(result);
      setSam2LearnerVideoVersion(Date.now());
      if (result.status === 'failed') {
        throw new Error(result.failure_reason || 'YOLO+SAM2 scissors tracking failed.');
      }
      if (toPlayableStorageUrl(result.overlay_video_url, result.overlay_video_path) || result.annotated_video_url) {
        setLearnerOverlay('sam2');
      }

      toast.success(
        `YOLO+SAM2 ready — ${result.device.toUpperCase()} • stride ${result.frame_stride}`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'YOLO+SAM2 scissors tracking failed.';
      setSam2LearnerError(message);
      toast.error(message);
    } finally {
      setIsSam2LearnerProcessing(false);
    }
  }, [userVideo]);

  // ── YOLO+Angle+DTW learner run ────────────────────────────────────────────
  const runAngleDtw = useCallback(async () => {
    if (!userVideo) {
      toast.error('Upload a practice video first');
      return;
    }

    const expertName = DEFAULT_SAM2_YOLO_EXPERT_CODE;

    setIsAngleProcessing(true);
    setAngleError(null);
    setAngleDtwResult(null);
    setAngleRunId(null);
    setAngleStatus('extracting');

    try {
      setAngleStatus('dtw');
      const result = await runLearnerAngle(userVideo, expertName);
      setAngleDtwResult(result.summary);
      setAngleRunId(result.run_id);
      setAngleStatus('done');
      toast.success('YOLO+Angle+DTW complete');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'YOLO+Angle+DTW pipeline failed.';
      setAngleError(message);
      setAngleStatus('idle');
      toast.error(message);
    } finally {
      setIsAngleProcessing(false);
    }
  }, [userVideo]);

  // ── SYNC DTW preview ───────────────────────────────────────────────────────
  const handleSyncToggleRequest = useCallback(
    (wantOn: boolean) => {
      if (!wantOn) {
        setIsSynced(false);
        setDtwPreviewUrl(null);
        return;
      }
      // Opening sync: show confirmation dialog instead of enabling immediately.
      setIsSyncDialogOpen(true);
    },
    [setIsSynced],
  );

  const handleSyncConfirm = useCallback(async () => {
    setIsSyncDialogOpen(false);
    if (!angleRunId) {
      toast.error('Run the YOLO+Angle+DTW pipeline first before enabling SYNC');
      return;
    }
    setIsDtwPreviewGenerating(true);
    try {
      const response = await generateDtwPreview(angleRunId);
      setDtwPreviewUrl(`http://localhost:8001/${response.preview_path}`);
      setIsSynced(true);
      toast.success('DTW aligned preview ready');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'DTW preview generation failed.';
      toast.error(message);
    } finally {
      setIsDtwPreviewGenerating(false);
    }
  }, [angleRunId, setIsSynced]);

  const runTipTracking = useCallback(
    async (tipX: number, tipY: number) => {
      if (!sam2LearnerRun) return;
      setIsTipTracking(true);
      setTipTrackingError(null);
      try {
        const response = await fetch(
          `/api/sam2/track-tip/${encodeURIComponent(sam2LearnerRun.run_id)}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              frame_index: 0,
              tip_point: [tipX, tipY],
            }),
          },
        );
        let payload: any = null;
        try {
          payload = await response.json();
        } catch {
          payload = null;
        }
        if (!response.ok) {
          throw new Error(
            payload?.detail?.message ||
              payload?.detail ||
              `Tip tracking failed with status ${response.status}`,
          );
        }
        const updated = payload as Sam2LearnerResult;
        setSam2LearnerRun(updated);
        setSam2LearnerVideoVersion(Date.now());
        setLearnerOverlay('sam2');
        setIsSelectingTip(false);
        toast.success('Tracked scissor tip extracted');
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Tip tracking failed.';
        setTipTrackingError(message);
        toast.error(message);
      } finally {
        setIsTipTracking(false);
      }
    },
    [sam2LearnerRun],
  );

  const handleLearnerTipClick = useCallback(
    (event: React.MouseEvent<HTMLElement>) => {
      // Game mark placement takes priority
      if (gamePhase === 'game' && activeMarkType) {
        if (!learnerVideoRef.current) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const video = learnerVideoRef.current;

        // Calculate actual video content area within the element (object-fit: contain)
        const videoAspect = video.videoWidth / video.videoHeight;
        const elementAspect = rect.width / rect.height;

        let contentWidth: number, contentHeight: number, contentLeft: number, contentTop: number;
        if (videoAspect > elementAspect) {
          // Black bars top and bottom
          contentWidth = rect.width;
          contentHeight = rect.width / videoAspect;
          contentLeft = 0;
          contentTop = (rect.height - contentHeight) / 2;
        } else {
          // Black bars left and right
          contentHeight = rect.height;
          contentWidth = rect.height * videoAspect;
          contentLeft = (rect.width - contentWidth) / 2;
          contentTop = 0;
        }

        // Raw element-relative position (used for visual display)
        const display_x = event.clientX - rect.left;
        const display_y = event.clientY - rect.top;

        // Content-relative position
        const content_x = display_x - contentLeft;
        const content_y = display_y - contentTop;

        // Ignore clicks in black bars
        if (content_x < 0 || content_y < 0 || content_x > contentWidth || content_y > contentHeight) return;

        // Pre-scale to native video pixel space so backend receives coordinates
        // already in the same space as the stored bounding boxes.
        const nativeW = video.videoWidth || contentWidth;
        const nativeH = video.videoHeight || contentHeight;
        const native_x = content_x * (nativeW / contentWidth);
        const native_y = content_y * (nativeH / contentHeight);

        setUserMarks((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            mark_type: activeMarkType,
            x: native_x,
            y: native_y,
            display_x,
            display_y,
            timestamp_sec: video.currentTime,
            video_width: nativeW,
            video_height: nativeH,
          },
        ]);
        return;
      }
      if (!isSelectingTip || !sam2LearnerRun || !learnerVideoRef.current) return;
      const video = learnerVideoRef.current;
      const rect = video.getBoundingClientRect();
      const nx = (event.clientX - rect.left) / rect.width;
      const ny = (event.clientY - rect.top) / rect.height;
      const x = Math.max(0, Math.min(1, nx)) * (video.videoWidth || sam2LearnerRun.metadata?.width || 1);
      const y = Math.max(0, Math.min(1, ny)) * (video.videoHeight || sam2LearnerRun.metadata?.height || 1);
      void runTipTracking(x, y);
    },
    [gamePhase, activeMarkType, isSelectingTip, sam2LearnerRun, runTipTracking],
  );

  const eventToVideoPixels = useCallback(
    (event: React.MouseEvent<HTMLDivElement>): [number, number] | null => {
      if (!learnerVideoRef.current) return null;
      const video = learnerVideoRef.current;
      const rect = video.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      const nx = (event.clientX - rect.left) / rect.width;
      const ny = (event.clientY - rect.top) / rect.height;
      const x = Math.max(0, Math.min(1, nx)) * (video.videoWidth || 1);
      const y = Math.max(0, Math.min(1, ny)) * (video.videoHeight || 1);
      return [x, y];
    },
    [],
  );

  const handleManualSamMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!isManualSamInitMode) return;
      const pt = eventToVideoPixels(event);
      if (!pt) return;
      manualSamDragStartRef.current = pt;
      setManualSamDraftBox([pt[0], pt[1], pt[0], pt[1]]);
    },
    [isManualSamInitMode, eventToVideoPixels],
  );

  const handleManualSamMouseMove = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const start = manualSamDragStartRef.current;
      if (!isManualSamInitMode || !start) return;
      const pt = eventToVideoPixels(event);
      if (!pt) return;
      setManualSamDraftBox([start[0], start[1], pt[0], pt[1]]);
    },
    [isManualSamInitMode, eventToVideoPixels],
  );

  const handleManualSamMouseUp = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const start = manualSamDragStartRef.current;
      manualSamDragStartRef.current = null;
      if (!isManualSamInitMode || !start) return;
      const pt = eventToVideoPixels(event);
      if (!pt) return;
      const dx = pt[0] - start[0];
      const dy = pt[1] - start[1];
      const dist = Math.hypot(dx, dy);
      if (dist > 10) {
        const box: [number, number, number, number] = [
          Math.min(start[0], pt[0]),
          Math.min(start[1], pt[1]),
          Math.max(start[0], pt[0]),
          Math.max(start[1], pt[1]),
        ];
        setManualSamPrompt((prev) => ({ ...prev, box }));
      } else {
        setManualSamPrompt((prev) => {
          if (manualSamPointMode === 'negative') {
            return {
              ...prev,
              negative_points: [...prev.negative_points, [pt[0], pt[1]]],
            };
          }
          return {
            ...prev,
            positive_points: [...prev.positive_points, [pt[0], pt[1]]],
          };
        });
      }
      setManualSamDraftBox(null);
    },
    [isManualSamInitMode, eventToVideoPixels, manualSamPointMode],
  );

  // ── Optical Flow run ─────────────────────────────────────────────────────

  const runOpticalFlow = useCallback(async () => {
    if (!userVideo) {
      toast.error('Upload a practice video first');
      return;
    }

    setIsOpticalFlowProcessing(true);
    setOpticalFlowError(null);

    try {
      const formData = new FormData();
      formData.append('file', userVideo);
      formData.append('save_visualization', 'true');
      formData.append('roi_source', 'yolo_scissors');
      formData.append('use_hand_roi', 'true');
      formData.append('roi_padding_px', '40');

      const response = await fetch('/api/optical-flow/learner', {
        method: 'POST',
        body: formData,
      });

      let payload: any = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail =
          payload?.detail ||
          payload?.message ||
          `Optical Flow failed with status ${response.status}`;
        throw new Error(detail);
      }

      const result = payload as OpticalFlowLearnerResult;
      setOpticalFlowRun(result);
      setOpticalFlowVideoVersion(Date.now());
      if (result.visualization_video_url || result.visualization_video_path) {
        setLearnerOverlay('optical_flow');
      }
      toast.success('Optical Flow ready');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Optical Flow processing failed.';
      setOpticalFlowError(message);
      toast.error(message);
    } finally {
      setIsOpticalFlowProcessing(false);
    }
  }, [userVideo]);

  // Compare Studio is learner-focused. The YOLO+SAM2 and Optical Flow tabs run
  // on the uploaded learner video and reuse precomputed expert artifacts by path.
  const mediapipeAnnotatedSource =
    mediapipeRun?.annotated_video_url && mediapipeVideoVersion > 0
      ? `${mediapipeRun.annotated_video_url}${mediapipeRun.annotated_video_url.includes('?') ? '&' : '?'}v=${mediapipeVideoVersion}`
      : mediapipeRun?.annotated_video_url ?? null;

  const sam2OverlayBaseUrl = toPlayableStorageUrl(
    sam2LearnerRun?.overlay_video_url,
    sam2LearnerRun?.overlay_video_path,
  );

  const corridorOverlayBaseUrl = toPlayableStorageUrl(
    sam2LearnerRun?.aligned_corridor_overlay_video_url,
    sam2LearnerRun?.aligned_corridor_overlay_video_path,
  );

  const sam2AnnotatedSource =
    (() => {
      const baseUrl =
        sam2OverlayBaseUrl ||
        sam2LearnerRun?.tip_annotated_video_url ||
        sam2LearnerRun?.annotated_video_url ||
        null;
      if (!baseUrl) return null;
      if (sam2LearnerVideoVersion > 0) {
        return `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}v=${sam2LearnerVideoVersion}`;
      }
      return baseUrl;
    })();

  useEffect(() => {
    setSam2OverlayVideoError(null);
  }, [sam2AnnotatedSource]);

  const opticalFlowVisualizationUrl =
    opticalFlowRun?.visualization_video_url ??
    storagePathToUrl(opticalFlowRun?.visualization_video_path);
  const opticalFlowLearnerSource =
    opticalFlowVisualizationUrl && opticalFlowVideoVersion > 0
      ? `${opticalFlowVisualizationUrl}${opticalFlowVisualizationUrl.includes('?') ? '&' : '?'}v=${opticalFlowVideoVersion}`
      : opticalFlowVisualizationUrl;

  const learnerVideoSource = (() => {
    if (learnerOverlay === 'mediapipe' && mediapipeAnnotatedSource) {
      return mediapipeAnnotatedSource;
    }
    if (learnerOverlay === 'sam2' && sam2AnnotatedSource) {
      return sam2AnnotatedSource;
    }
    if (learnerOverlay === 'optical_flow' && opticalFlowLearnerSource) {
      return opticalFlowLearnerSource;
    }
    if (learnerOverlay === 'aligned_corridor' && corridorOverlayBaseUrl) {
      return corridorOverlayBaseUrl;
    }
    if (learnerOverlay === 'eval_corridor' && evalCorridorOverlayUrl) {
      return evalCorridorOverlayUrl;
    }
    if (learnerOverlay === 'visualization' && vizUrl) {
      return vizUrl;
    }
    return userVideoUrl;
  })();
  const learnerVideoPixelWidth =
    learnerVideoRef.current?.videoWidth || sam2LearnerRun?.metadata?.width || 1;
  const learnerVideoPixelHeight =
    learnerVideoRef.current?.videoHeight || sam2LearnerRun?.metadata?.height || 1;

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
        <div
          style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}
          title="⚠️ Generating aligned preview may take up to 7 minutes"
        >
          <span className="label">Sync</span>
          <Switch checked={isSynced} onCheckedChange={handleSyncToggleRequest} />
          {isDtwPreviewGenerating && (
            <Loader2 size={14} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }} />
          )}
        </div>

        {/* ── SYNC confirmation dialog ───────────────────────────────────── */}
        {isSyncDialogOpen && (
          <div
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 9999,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(0,0,0,0.55)',
            }}
            onClick={() => setIsSyncDialogOpen(false)}
          >
            <div
              className="card"
              style={{
                padding: 'var(--space-xl)',
                maxWidth: 420,
                width: '90vw',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-md)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-sm)' }}>
                <span style={{ fontSize: '1.5rem', lineHeight: 1 }}>⚠️</span>
                <div>
                  <p style={{ margin: 0, fontWeight: 700, fontSize: '1rem' }}>
                    Generate DTW Aligned Preview?
                  </p>
                  <p className="text-small" style={{ color: 'var(--text-muted)', marginTop: 6 }}>
                    Generating the DTW aligned preview may take up to 7 minutes depending on
                    video length. This will run the preview generation pipeline in the background.
                  </p>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => setIsSyncDialogOpen(false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void handleSyncConfirm()}
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Path Overlay button ──────────────────────────────────────── */}
        <button
          className={`btn ${pathOverlayState === 'ready' ? 'btn-primary' : 'btn-secondary'}`}
          title={pathOverlayState === 'disabled' ? 'Run evaluation first' : undefined}
          disabled={pathOverlayState === 'disabled' || pathOverlayState === 'loading'}
          style={pathOverlayState === 'disabled' ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
          onClick={async () => {
            if (pathOverlayState !== 'idle') return;
            if (!evalEvaluationId || !evalRunId) return;
            setPathOverlayState('loading');
            console.log('[PATH OVERLAY] evaluation_id:', evalEvaluationId);
            console.log('[PATH OVERLAY] run_id being sent:', evalRunId);
            try {
              const res = await fetch(
                `/api/evaluations/${evalEvaluationId}/generate-corridor-overlay`,
                {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ run_id: evalRunId }),
                },
              );
              if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error((err as any).detail ?? `HTTP ${res.status}`);
              }
              const data = await res.json() as { status: string; overlay_video_url: string };
              const fullUrl = data.overlay_video_url.startsWith('http')
                ? data.overlay_video_url
                : `http://localhost:8001${data.overlay_video_url}`;
              setEvalCorridorOverlayUrl(fullUrl);
              setPathOverlayState('ready');
            } catch (err) {
              setPathOverlayState('idle');
              console.error('Path overlay generation failed:', err);
            }
          }}
        >
          {pathOverlayState === 'loading' ? (
            <>
              <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
              Generating...
            </>
          ) : (
            'Path Overlay'
          )}
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

          {expertVideoUrl ? (
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
                  key={dtwPreviewUrl ?? expertVideoUrl ?? 'missing-expert-video'}
                  src={dtwPreviewUrl ?? expertVideoUrl ?? undefined}
                  poster={dtwPreviewUrl ? undefined : clip?.thumbnail}
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
                  {hasSelectedExpertReference ? 'Expert Video Unavailable' : 'Choose An Expert Video'}
                </p>
                <p className="empty-state-description">
                  {hasSelectedExpertReference
                    ? (expertVideoError ?? 'Loading expert video for the selected chapter...')
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
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
                  {(vizUrl ||
                  mediapipeRun?.annotated_video_url ||
                  sam2OverlayBaseUrl ||
                  sam2LearnerRun?.annotated_video_url ||
                  corridorOverlayBaseUrl ||
                  evalCorridorOverlayUrl ||
                  opticalFlowVisualizationUrl) &&
                  !(gamePhase === 'game' && activeMarkType) && (
                  <div
                    role="tablist"
                    aria-label="Learner video source"
                    style={{
                      display: 'inline-flex',
                      border: '1px solid var(--border-default)',
                      borderRadius: 'var(--radius-sm)',
                      overflow: 'hidden',
                    }}
                  >
                    <button
                      type="button"
                      role="tab"
                      aria-selected={learnerOverlay === 'none'}
                      className={`btn ${learnerOverlay === 'none' ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                      onClick={() => setLearnerOverlay('none')}
                    >
                      Original
                    </button>
                    {mediapipeRun?.annotated_video_url && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'mediapipe'}
                        className={`btn ${learnerOverlay === 'mediapipe' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay('mediapipe')}
                      >
                        <Activity size={12} style={{ marginRight: 4 }} />
                        MediaPipe
                      </button>
                    )}
                    {(sam2OverlayBaseUrl || sam2LearnerRun?.annotated_video_url) && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'sam2'}
                        className={`btn ${learnerOverlay === 'sam2' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay('sam2')}
                      >
                        <Hand size={12} style={{ marginRight: 4 }} />
                        YOLO+SAM2 Scissors
                      </button>
                    )}
                    {corridorOverlayBaseUrl && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'aligned_corridor'}
                        className={`btn ${learnerOverlay === 'aligned_corridor' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay('aligned_corridor')}
                      >
                        <Hand size={12} style={{ marginRight: 4 }} />
                        Aligned Expert Corridor
                      </button>
                    )}
                    {evalCorridorOverlayUrl && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'eval_corridor'}
                        className={`btn ${learnerOverlay === 'eval_corridor' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay(learnerOverlay === 'eval_corridor' ? 'none' : 'eval_corridor')}
                      >
                        <Activity size={12} style={{ marginRight: 4 }} />
                        Path Overlay
                      </button>
                    )}
                    {console.log('vizUrl state:', vizUrl, 'gamePhase:', gamePhase, 'evalPhase:', evalPhase) as undefined}
                    {vizUrl && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'visualization'}
                        className={`btn ${learnerOverlay === 'visualization' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay('visualization')}
                      >
                        Visualization
                      </button>
                    )}
                    {opticalFlowVisualizationUrl && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'optical_flow'}
                        className={`btn ${learnerOverlay === 'optical_flow' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay('optical_flow')}
                      >
                        <Activity size={12} style={{ marginRight: 4 }} />
                        Flow
                      </button>
                    )}
                    {angleDtwResult && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'angle'}
                        className={`btn ${learnerOverlay === 'angle' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay((cur) => cur === 'angle' ? 'none' : 'angle')}
                      >
                        <Triangle size={12} style={{ marginRight: 4 }} />
                        Angle Overlay
                      </button>
                    )}
                  </div>
                )}
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
              </div>
            ) : null}
          </div>

          {userVideoUrl ? (
            <>
              <div
                className="video-container"
                style={{
                  aspectRatio: '16/9',
                  marginBottom: 'var(--space-sm)',
                  position: 'relative',
                }}
              >
                <video
                  key={learnerVideoSource ?? userVideoUrl}
                  ref={learnerVideoRef}
                  src={learnerVideoSource ?? undefined}
                  controls={(learnerOverlay === 'sam2' || learnerOverlay === 'optical_flow' || learnerOverlay === 'aligned_corridor' || learnerOverlay === 'eval_corridor' || learnerOverlay === 'visualization') && gamePhase !== 'game'}
                  onClick={handleLearnerTipClick}
                  muted={learnerMuted}
                  playsInline
                  preload="auto"
                  onTimeUpdate={() => {
                    if (learnerVideoRef.current)
                      setLearnerCurrentTime(learnerVideoRef.current.currentTime);
                  }}
                  onLoadedMetadata={() => {
                    if (learnerVideoRef.current)
                      setLearnerDuration(learnerVideoRef.current.duration);
                  }}
                  onError={() => {
                    if (learnerOverlay === 'sam2') {
                      console.error('YOLO+SAM2 overlay video failed to load:', learnerVideoSource);
                      setSam2OverlayVideoError('YOLO+SAM2 overlay video could not be loaded.');
                    }
                    if (learnerOverlay === 'optical_flow') {
                      setOpticalFlowError(
                        'Optical Flow video was generated, but the browser could not load it.',
                      );
                    }
                  }}
                  style={{
                    cursor: (gamePhase === 'game' && activeMarkType) || isSelectingTip ? 'crosshair' : 'default',
                    pointerEvents: (gamePhase === 'game' && activeMarkType) ? 'none' : 'auto'
                  }}
                />

                {gamePhase === 'game' && activeMarkType && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      zIndex: 20,
                      background: 'transparent',
                      cursor: 'crosshair',
                      pointerEvents: 'all',
                    }}
                    onClick={handleLearnerTipClick}
                  />
                )}

                {/* DTW preview loading overlay */}
                {isDtwPreviewGenerating && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 'var(--space-sm)',
                      background: 'rgba(0,0,0,0.6)',
                      borderRadius: 'inherit',
                    }}
                  >
                    <Loader2
                      size={32}
                      style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }}
                    />
                    <span
                      className="text-small"
                      style={{ color: '#fff', fontWeight: 600, textAlign: 'center', padding: '0 var(--space-md)' }}
                    >
                      Generating DTW aligned preview…
                      <br />
                      <span style={{ fontWeight: 400, opacity: 0.8 }}>This may take up to 7 minutes.</span>
                    </span>
                  </div>
                )}

                {/* ── Game marks overlay ───────────────────────────────── */}
                {(gamePhase === 'game' || gamePhase === 'result') && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      pointerEvents: 'none',
                      overflow: 'hidden',
                    }}
                  >
                    {/* User-placed marks */}
                    {userMarks.map((mark, markIdx) => {
                      const resultEntry = scoreResult?.mark_results[markIdx];
                      const isCorrect = resultEntry?.result === 'correct';
                      const isFalseAlarm = resultEntry?.result === 'false_alarm';
                      const pendingColor =
                        mark.mark_type === 'vibration' ? '#F59E0B' : '#ef4444';
                      const color =
                        gamePhase === 'result'
                          ? isCorrect
                            ? '#22c55e'
                            : '#ef4444'
                          : pendingColor;
                      const symbol =
                        mark.mark_type === 'trajectory' ? '✕'
                        : mark.mark_type === 'angle' ? '○'
                        : '〜';
                      const fSize =
                        mark.mark_type === 'trajectory' ? 28
                        : mark.mark_type === 'vibration' ? 30
                        : 36;
                      return (
                        <span
                          key={mark.id}
                          title={gamePhase === 'result' ? (isCorrect ? 'Correct!' : 'False alarm') : 'Click to remove'}
                          onClick={
                            gamePhase === 'game'
                              ? () => setUserMarks((prev) => prev.filter((m) => m.id !== mark.id))
                              : undefined
                          }
                          style={{
                            position: 'absolute',
                            left: mark.display_x,
                            top: mark.display_y,
                            transform: 'translate(-50%, -50%)',
                            fontSize: fSize,
                            lineHeight: 1,
                            color,
                            textShadow: '0 0 3px #fff, 0 0 6px #fff',
                            pointerEvents: gamePhase === 'game' ? 'auto' : 'none',
                            cursor: gamePhase === 'game' ? 'pointer' : 'default',
                            userSelect: 'none',
                            width: 40,
                            height: 40,
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            textDecoration: isFalseAlarm && gamePhase === 'result' ? 'line-through' : 'none',
                          }}
                        >
                          {symbol}
                        </span>
                      );
                    })}

                    {/* Missed errors shown in result phase */}
                    {gamePhase === 'result' &&
                      scoreResult?.missed_error_details.map((e) => {
                        const video = learnerVideoRef.current;
                        if (!video) return null;

                        const vw = video.offsetWidth;
                        const vh = video.offsetHeight;
                        if (!vw || !vh) return null;

                        // Letterbox calculation (same as handleLearnerTipClick)
                        const videoAspect = video.videoWidth / video.videoHeight;
                        const elementAspect = vw / vh;

                        let contentWidth: number, contentHeight: number, contentLeft: number, contentTop: number;
                        if (videoAspect > elementAspect) {
                          contentWidth = vw;
                          contentHeight = vw / videoAspect;
                          contentLeft = 0;
                          contentTop = (vh - contentHeight) / 2;
                        } else {
                          contentHeight = vh;
                          contentWidth = vh * videoAspect;
                          contentLeft = (vw - contentWidth) / 2;
                          contentTop = 0;
                        }

                        // Scale from native video coords to display coords
                        const nativeW = video.videoWidth || 1440;
                        const nativeH = video.videoHeight || 1080;
                        const dx = contentLeft + (e.peak_location.x / nativeW) * contentWidth;
                        const dy = contentTop + (e.peak_location.y / nativeH) * contentHeight;
                        const missedSymbol =
                          e.error_type === 'trajectory' ? '✕'
                          : e.error_type === 'vibration' ? '〜'
                          : '○';
                        const missedSize =
                          e.error_type === 'trajectory' ? 28
                          : e.error_type === 'vibration' ? 30
                          : 36;
                        return (
                          <span
                            key={`missed-${e.error_id}`}
                            title="Missed"
                            style={{
                              position: 'absolute',
                              left: dx,
                              top: dy,
                              transform: 'translate(-50%, -50%)',
                              fontSize: missedSize,
                              lineHeight: 1,
                              color: 'rgba(156,163,175,0.85)',
                              textShadow: '0 0 3px #000',
                              pointerEvents: 'none',
                              userSelect: 'none',
                              display: 'inline-flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              gap: 2,
                            }}
                          >
                            {missedSymbol}
                            <span style={{ fontSize: 9, fontWeight: 600, color: 'rgba(209,213,219,0.9)', lineHeight: 1 }}>
                              missed
                            </span>
                          </span>
                        );
                      })}
                  </div>
                )}

                {(isManualSamInitMode ||
                  ((manualSamPrompt.box ||
                    manualSamPrompt.positive_points.length > 0 ||
                    manualSamPrompt.negative_points.length > 0) &&
                    learnerCurrentTime <= 0.08)) && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      cursor: isManualSamInitMode ? 'crosshair' : 'default',
                    }}
                    onMouseDown={handleManualSamMouseDown}
                    onMouseMove={handleManualSamMouseMove}
                    onMouseUp={handleManualSamMouseUp}
                  >
                    {manualSamPrompt.box && (
                      <div
                        style={{
                          position: 'absolute',
                          left: `${(manualSamPrompt.box[0] / learnerVideoPixelWidth) * 100}%`,
                          top: `${(manualSamPrompt.box[1] / learnerVideoPixelHeight) * 100}%`,
                          width: `${((manualSamPrompt.box[2] - manualSamPrompt.box[0]) / learnerVideoPixelWidth) * 100}%`,
                          height: `${((manualSamPrompt.box[3] - manualSamPrompt.box[1]) / learnerVideoPixelHeight) * 100}%`,
                          border: '2px solid #F59E0B',
                          background: 'rgba(245,158,11,0.12)',
                        }}
                      />
                    )}
                    {manualSamDraftBox && (
                      <div
                        style={{
                          position: 'absolute',
                          left: `${(Math.min(manualSamDraftBox[0], manualSamDraftBox[2]) / learnerVideoPixelWidth) * 100}%`,
                          top: `${(Math.min(manualSamDraftBox[1], manualSamDraftBox[3]) / learnerVideoPixelHeight) * 100}%`,
                          width: `${(Math.abs(manualSamDraftBox[2] - manualSamDraftBox[0]) / learnerVideoPixelWidth) * 100}%`,
                          height: `${(Math.abs(manualSamDraftBox[3] - manualSamDraftBox[1]) / learnerVideoPixelHeight) * 100}%`,
                          border: '2px dashed #FBBF24',
                          background: 'rgba(251,191,36,0.08)',
                        }}
                      />
                    )}
                    {manualSamPrompt.positive_points.map((pt, idx) => (
                      <div
                        key={`sam-pos-${idx}`}
                        style={{
                          position: 'absolute',
                          left: `${(pt[0] / learnerVideoPixelWidth) * 100}%`,
                          top: `${(pt[1] / learnerVideoPixelHeight) * 100}%`,
                          width: 10,
                          height: 10,
                          borderRadius: '50%',
                          background: '#22c55e',
                          transform: 'translate(-50%, -50%)',
                          border: '1px solid #000',
                        }}
                      />
                    ))}
                    {manualSamPrompt.negative_points.map((pt, idx) => (
                      <div
                        key={`sam-neg-${idx}`}
                        style={{
                          position: 'absolute',
                          left: `${(pt[0] / learnerVideoPixelWidth) * 100}%`,
                          top: `${(pt[1] / learnerVideoPixelHeight) * 100}%`,
                          width: 10,
                          height: 10,
                          borderRadius: '50%',
                          background: '#ef4444',
                          transform: 'translate(-50%, -50%)',
                          border: '1px solid #000',
                        }}
                      />
                    ))}
                  </div>
                )}
                {learnerOverlay === 'mediapipe' && mediapipeRun?.annotated_video_url && (
                  <span
                    className="badge badge-blue"
                    style={{
                      position: 'absolute',
                      top: 8,
                      left: 8,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      fontSize: '0.65rem',
                    }}
                  >
                    <Activity size={10} />
                    MediaPipe
                  </span>
                )}
                {learnerOverlay === 'sam2' &&
                  (sam2OverlayBaseUrl || sam2LearnerRun?.annotated_video_url) && (
                  <span
                    className="badge badge-green"
                    style={{
                      position: 'absolute',
                      top: 8,
                      left: 8,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      fontSize: '0.65rem',
                    }}
                  >
                    <Hand size={10} />
                    YOLO+SAM2 Scissors
                  </span>
                )}
                {learnerOverlay === 'aligned_corridor' && corridorOverlayBaseUrl && (
                  <span
                    className="badge badge-green"
                    style={{
                      position: 'absolute',
                      top: 8,
                      left: 8,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      fontSize: '0.65rem',
                    }}
                  >
                    <Hand size={10} />
                    Aligned Expert Corridor
                  </span>
                )}
                {learnerOverlay === 'sam2' && sam2OverlayVideoError && (
                  <div
                    className="glass"
                    style={{
                      position: 'absolute',
                      left: '50%',
                      bottom: 12,
                      transform: 'translateX(-50%)',
                      padding: 'var(--space-xs) var(--space-sm)',
                      color: 'var(--accent-danger)',
                      fontSize: '0.75rem',
                      maxWidth: '90%',
                      textAlign: 'center',
                    }}
                  >
                    {sam2OverlayVideoError}
                  </div>
                )}
                {isSelectingTip && (
                  <span
                    className="badge badge-blue"
                    style={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      fontSize: '0.65rem',
                    }}
                  >
                    Click scissor tip on frame 0
                  </span>
                )}
                {isManualSamInitMode && (
                  <span
                    className="badge badge-blue"
                    style={{
                      position: 'absolute',
                      top: 32,
                      right: 8,
                      fontSize: '0.65rem',
                    }}
                  >
                    SAM2 Manual Init (frame 0)
                  </span>
                )}
                {learnerOverlay === 'optical_flow' && opticalFlowVisualizationUrl && (
                  <span
                    className="badge badge-blue"
                    style={{
                      position: 'absolute',
                      top: 8,
                      left: 8,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      fontSize: '0.65rem',
                    }}
                  >
                    <Activity size={10} />
                    Optical Flow
                  </span>
                )}
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
          <Tabs defaultValue="evaluate">
            <TabsList>
              <TabsTrigger value="evaluate">Evaluate</TabsTrigger>
              <TabsTrigger value="mediapipe">Models (Developer Tab)</TabsTrigger>
              <TabsTrigger value="timers">Timers</TabsTrigger>
            </TabsList>

            {/* ── Tab: Evaluate ─────────────────────────────────────────── */}
            <TabsContent value="evaluate">

              {/* ── Idle: run button ─────────────────────────────────────── */}
              {gamePhase === 'idle' && evalPhase === 'idle' && !apiEvaluationResult && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-lg)',
                    padding: 'var(--space-lg) 0',
                  }}
                >
                  <Sparkles size={48} style={{ color: 'var(--text-muted)' }} />
                  <p className="text-body" style={{ textAlign: 'center' }}>
                    {userVideo
                      ? 'Ready to evaluate your practice!'
                      : 'Upload a practice video to get started.'}
                  </p>
                  <button
                    className="btn btn-primary"
                    style={{ width: '100%' }}
                    disabled={!userVideo || !selectedClip}
                    onClick={() => void runEvaluation()}
                  >
                    Run Evaluation
                  </button>
                </div>
              )}

              {/* ── Progress panel (uploading / streaming) ───────────────── */}
              {(evalPhase === 'uploading' || evalPhase === 'streaming') && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-md)',
                    marginTop: 'var(--space-sm)',
                  }}
                >
                  {/* Header */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                    <Sparkles size={18} style={{ color: 'var(--accent-primary)' }} />
                    <span className="text-small" style={{ fontWeight: 600 }}>
                      Evaluating your practice
                    </span>
                  </div>

                  {/* Vertical stepper */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 6,
                    }}
                  >
                    {EVAL_STEPS.map((label, i) => {
                      const isDone    = i < evalStepIndex;
                      const isRunning = i === evalStepIndex;
                      const isWaiting = i > evalStepIndex;
                      return (
                        <div
                          key={label}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'var(--space-sm)',
                            padding: '5px 0',
                          }}
                        >
                          {/* Icon */}
                          <span
                            style={{
                              width: 18,
                              textAlign: 'center',
                              fontSize: '0.8rem',
                              flexShrink: 0,
                              color: isDone
                                ? 'var(--success, #22c55e)'
                                : isRunning
                                  ? 'var(--accent-primary)'
                                  : 'var(--text-muted)',
                            }}
                          >
                            {isDone ? '✓' : isRunning ? (
                              <Loader2
                                size={13}
                                style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}
                              />
                            ) : '○'}
                          </span>

                          {/* Label */}
                          <span
                            className="text-small"
                            style={{
                              color: isDone
                                ? 'var(--text-secondary)'
                                : isRunning
                                  ? 'var(--text-primary)'
                                  : 'var(--text-muted)',
                              fontWeight: isRunning ? 600 : 400,
                              opacity: isWaiting ? 0.5 : 1,
                            }}
                          >
                            {label}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Progress bar */}
                  <Progress value={evalProgress} />

                  {/* Rotating hint */}
                  <p
                    className="text-small"
                    style={{
                      color: 'var(--text-muted)',
                      textAlign: 'center',
                      margin: 0,
                      minHeight: '1.2em',
                    }}
                  >
                    {EVAL_HINTS[evalHintIndex]}
                  </p>
                </div>
              )}

              {/* ── Game: mark the errors ────────────────────────────────── */}
              {gamePhase === 'game' && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-md)',
                    padding: 'var(--space-sm) 0',
                  }}
                >
                  <div>
                    <p className="text-body" style={{ fontWeight: 600, marginBottom: 2 }}>
                      Mark the errors you spotted
                    </p>
                    <p className="text-small" style={{ color: 'var(--text-muted)', margin: 0 }}>
                      Watch the video and click where you noticed mistakes
                    </p>
                  </div>

                  {/* Toggle buttons */}
                  <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
                    {(['trajectory', 'angle', 'vibration'] as const).map((type) => {
                      const isActive = activeMarkType === type;
                      const isVib = type === 'vibration';
                      const symbol = type === 'trajectory' ? '✕' : type === 'angle' ? '○' : '〜';
                      const label  = type === 'trajectory' ? 'Path' : type === 'angle' ? 'Angle' : 'Vibration';
                      return (
                        <button
                          key={type}
                          className={isActive ? 'btn btn-primary' : 'btn btn-secondary'}
                          style={{
                            flex: 1,
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 6,
                            ...(isVib && isActive  ? { background: '#D97706', borderColor: '#B45309', color: '#fff' } : {}),
                            ...(isVib && !isActive ? { borderColor: '#D97706', color: '#F59E0B' } : {}),
                          }}
                          onClick={() => setActiveMarkType((prev) => (prev === type ? null : type))}
                        >
                          <span style={{ fontSize: 16 }}>{symbol}</span>
                          {label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Mark counts */}
                  <div className="text-small" style={{ color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <span>Marks placed: {userMarks.filter((m) => m.mark_type === 'trajectory').length} trajectory</span>
                    <span style={{ paddingLeft: 70 }}>{userMarks.filter((m) => m.mark_type === 'angle').length} angle</span>
                    <span style={{ paddingLeft: 70, color: '#F59E0B' }}>{userMarks.filter((m) => m.mark_type === 'vibration').length} vibration</span>
                  </div>

                  <hr style={{ border: 'none', borderTop: '1px solid var(--border-subtle)', margin: 0 }} />

                  <button
                    className="btn btn-primary"
                    style={{ width: '100%' }}
                    disabled={userMarks.length === 0}
                    onClick={async () => {
                      if (!evalEvaluationId || !evalRunId) return;
                      try {
                        const res = await fetch(`/api/evaluations/${evalEvaluationId}/score`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ run_id: evalRunId, marks: userMarks }),
                        });
                        if (!res.ok) throw new Error(`Score API error: ${res.status}`);
                        const data = await res.json() as ScoreResult;
                        setScoreResult(data);
                        setGamePhase('result');
                      } catch (err) {
                        toast.error('Failed to score your marks. Please try again.');
                        console.error(err);
                      }
                    }}
                  >
                    Get Your Result
                  </button>

                  <button
                    className="btn btn-secondary"
                    style={{ width: '100%' }}
                    onClick={() => {
                      sessionStorage.removeItem('augmentor_compare_state');
                      setGamePhase('idle');
                      setEvalPhase('idle');
                      setEvalStepIndex(0);
                      setEvalProgress(0);
                      setEvalError(null);
                      setActiveMarkType(null);
                      setUserMarks([]);
                      setScoreResult(null);
                      setGameErrors([]);
                      setFeedbackText(null);
                      setFeedbackStatus('idle');
                    }}
                  >
                    <RotateCcw size={14} />
                    Run Again
                  </button>
                </div>
              )}

              {/* ── Result: score display ─────────────────────────────────── */}
              {gamePhase === 'result' && scoreResult && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-md)',
                    padding: 'var(--space-sm) 0',
                  }}
                >
                  <p className="text-body" style={{ fontWeight: 600, margin: 0 }}>Your Score</p>

                  {/* Big score */}
                  <div style={{ textAlign: 'center', padding: 'var(--space-sm) 0' }}>
                    <span
                      style={{
                        fontSize: '3rem',
                        fontWeight: 700,
                        color: 'var(--accent-primary)',
                        lineHeight: 1,
                      }}
                    >
                      {scoreResult.score_pct}%
                    </span>
                  </div>

                  {/* Summary rows */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <div className="text-small" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: '#22c55e', fontWeight: 700, width: 14, textAlign: 'center' }}>✓</span>
                      <span>{scoreResult.correct_marks} error{scoreResult.correct_marks !== 1 ? 's' : ''} found</span>
                    </div>
                    <div className="text-small" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: '#ef4444', fontWeight: 700, width: 14, textAlign: 'center' }}>✕</span>
                      <span>{scoreResult.false_alarms} false alarm{scoreResult.false_alarms !== 1 ? 's' : ''}</span>
                    </div>
                    <div className="text-small" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: 'var(--text-muted)', width: 14, textAlign: 'center' }}>○</span>
                      <span>{scoreResult.missed_errors} error{scoreResult.missed_errors !== 1 ? 's' : ''} missed</span>
                    </div>
                  </div>

                  <hr style={{ border: 'none', borderTop: '1px solid var(--border-subtle)', margin: 0 }} />

                  {/* Breakdown by type */}
                  {(['trajectory', 'angle', 'vibration'] as const).map((type) => {
                    const totalReal = scoreResult.missed_error_details.filter((e) => e.error_type === type).length
                      + scoreResult.mark_results.filter((m) => m.mark_type === type && m.result === 'correct').length;
                    const found = scoreResult.mark_results.filter((m) => m.mark_type === type && m.result === 'correct').length;
                    if (totalReal === 0 && found === 0) return null;
                    const label = type === 'trajectory' ? 'Trajectory' : type === 'angle' ? 'Angle' : 'Vibration';
                    const labelColor = type === 'vibration' ? '#F59E0B' : 'var(--text-secondary)';
                    return (
                      <div
                        key={type}
                        className="text-small"
                        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                      >
                        <span style={{ color: labelColor }}>{label}</span>
                        <span style={{ color: 'var(--text-muted)' }}>
                          {totalReal} real &nbsp;|&nbsp; {found} found
                        </span>
                      </div>
                    );
                  })}

                  {/* Vibration error detail list */}
                  {gameErrors.filter((e) => e.error_type === 'vibration').length > 0 && (
                    <>
                      <hr style={{ border: 'none', borderTop: '1px solid var(--border-subtle)', margin: 0 }} />
                      <div>
                        <p
                          className="text-small"
                          style={{ fontWeight: 600, color: '#F59E0B', marginBottom: 6, margin: '0 0 6px' }}
                        >
                          Vibration Events
                        </p>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                          {gameErrors
                            .filter((e) => e.error_type === 'vibration')
                            .map((e) => (
                              <div
                                key={`vib-detail-${e.error_id}`}
                                className="text-small"
                                style={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: 2,
                                  padding: '6px 8px',
                                  background: 'rgba(245,158,11,0.08)',
                                  border: '1px solid rgba(245,158,11,0.25)',
                                  borderRadius: 6,
                                }}
                              >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <span style={{ color: '#F59E0B', fontWeight: 600 }}>
                                    〜 Vibration #{e.error_id}
                                  </span>
                                  {e.severity && (
                                    <span
                                      style={{
                                        fontSize: 10,
                                        fontWeight: 700,
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.04em',
                                        color:
                                          e.severity === 'severe' ? '#ef4444'
                                          : e.severity === 'moderate' ? '#F59E0B'
                                          : '#a3a3a3',
                                      }}
                                    >
                                      {e.severity}
                                    </span>
                                  )}
                                </div>
                                <div style={{ color: 'var(--text-muted)', display: 'flex', gap: 12 }}>
                                  <span>
                                    {e.timestamp_start_sec.toFixed(1)}s – {e.timestamp_end_sec.toFixed(1)}s
                                  </span>
                                  {e.dominant_freq_hz != null && (
                                    <span>{e.dominant_freq_hz.toFixed(1)} Hz</span>
                                  )}
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    </>
                  )}

                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', marginTop: 'var(--space-xs)' }}
                    onClick={() => {
                      sessionStorage.removeItem('augmentor_compare_state');
                      setGamePhase('idle');
                      setEvalPhase('idle');
                      setEvalStepIndex(0);
                      setEvalProgress(0);
                      setEvalError(null);
                      setActiveMarkType(null);
                      setUserMarks([]);
                      setScoreResult(null);
                      setGameErrors([]);
                      setFeedbackText(null);
                      setFeedbackStatus('idle');
                    }}
                  >
                    <RotateCcw size={14} />
                    Run Again
                  </button>

                  <button
                    className={`btn ${vizState === 'ready' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ width: '100%' }}
                    disabled={vizState === 'loading'}
                    onClick={async () => {
                      if (vizState !== 'idle') return;
                      if (!evalEvaluationId || !evalRunId || !selectedClip) return;
                      setVizState('loading');
                      try {
                        const res = await fetch(
                          `/api/evaluations/${evalEvaluationId}/generate-visualization`,
                          {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ run_id: evalRunId, expert_id: selectedClip }),
                          },
                        );
                        if (!res.ok) {
                          const err = await res.json().catch(() => ({}));
                          throw new Error((err as any).detail ?? `HTTP ${res.status}`);
                        }
                        const data = await res.json() as { status: string; visualization_url: string };
                        const fullUrl = data.visualization_url.startsWith('http')
                          ? data.visualization_url
                          : `http://localhost:8001${data.visualization_url}`;
                        setVizUrl(fullUrl);
                        console.log('vizUrl set to:', fullUrl);
                        setVizState('ready');
                      } catch (err) {
                        setVizState('idle');
                        console.error('Visualization generation failed:', err);
                      }
                    }}
                  >
                    {vizState === 'loading' ? (
                      <>
                        <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                        Generating...
                      </>
                    ) : (
                      'Visualization'
                    )}
                  </button>
                </div>
              )}

              {/* ── Error ────────────────────────────────────────────────── */}
              {evalPhase === 'error' && (
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
                      width: 56,
                      height: 56,
                      borderRadius: '50%',
                      border: '3px solid #ef4444',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 0 16px #ef444430',
                    }}
                  >
                    <span style={{ fontSize: '1.5rem', lineHeight: 1 }}>!</span>
                  </div>
                  <p
                    className="text-small"
                    style={{ textAlign: 'center', color: 'var(--text-secondary)', maxWidth: 280 }}
                  >
                    {evalError ?? 'Something went wrong. Please try again.'}
                  </p>
                  <button
                    className="btn btn-secondary"
                    style={{ width: '100%' }}
                    onClick={() => {
                      sessionStorage.removeItem('augmentor_compare_state');
                      setEvalPhase('idle');
                      setEvalStepIndex(0);
                      setEvalProgress(0);
                      setEvalError(null);
                      setGamePhase('idle');
                    }}
                  >
                    <RotateCcw size={14} />
                    Try Again
                  </button>
                </div>
              )}

              {/* ── Out-of-context rejection (existing flow) ─────────────── */}
              {gamePhase === 'idle' && evalPhase === 'idle' && apiEvaluationResult?.status === 'out_of_context' && (
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
                  <p className="heading-4" style={{ textAlign: 'center', color: '#ef4444' }}>
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
                      sessionStorage.removeItem('augmentor_compare_state');
                      resetEvaluation();
                      setApiEvaluationResult(null);
                    }}
                  >
                    <RotateCcw size={14} />
                    Try Again
                  </button>
                </div>
              )}

            </TabsContent>

            {/* ── Tab: MediaPipe ───────────────────────────────────────── */}
            <TabsContent value="mediapipe">
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 'var(--space-md)',
                  padding: 'var(--space-sm) 0',
                }}
              >
                <Tabs value={inspectionModel} onValueChange={(v) => setInspectionModel(v as InspectionModel)}>
                  <TabsList>
                    <TabsTrigger value="mediapipe">MediaPipe</TabsTrigger>
                    <TabsTrigger value="sam2">YOLO+SAM2 Scissors</TabsTrigger>
                    <TabsTrigger value="optical_flow">Optical Flow</TabsTrigger>
                    <TabsTrigger value="yolo_angle">YOLO+Angle+DTW</TabsTrigger>
                  </TabsList>

                  <TabsContent value="mediapipe">
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-md)',
                        marginTop: 'var(--space-sm)',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 'var(--space-sm)',
                        }}
                      >
                        <Activity size={18} style={{ color: 'var(--accent-primary)' }} />
                        <span className="text-small" style={{ fontWeight: 600 }}>
                          Hand Tracking (MediaPipe)
                        </span>
                      </div>
                      <p
                        className="text-small"
                        style={{ color: 'var(--text-muted)', margin: 0 }}
                      >
                        Run MediaPipe on your practice video to produce an annotated
                        overlay with hand landmarks, wrist trajectory and bounding
                        box. Toggle between the original and the annotated version
                        above the learner video.
                      </p>

                      <button
                        className="btn btn-primary"
                        style={{
                          width: '100%',
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 'var(--space-xs)',
                        }}
                        disabled={!userVideo || isMediapipeProcessing}
                        onClick={runMediapipe}
                      >
                        {isMediapipeProcessing ? (
                          <>
                            <Loader2
                              size={14}
                              style={{ animation: 'spin 1s linear infinite' }}
                            />
                            Processing...
                          </>
                        ) : (
                          <>
                            <Activity size={14} />
                            {mediapipeRun ? 'Run Again' : 'Run MediaPipe'}
                          </>
                        )}
                      </button>

                      {!userVideo && (
                        <p
                          className="text-small"
                          style={{
                            color: 'var(--text-muted)',
                            textAlign: 'center',
                            margin: 0,
                          }}
                        >
                          Upload a practice video to enable MediaPipe.
                        </p>
                      )}

                      {mediapipeError && (
                        <div
                          className="text-small"
                          style={{
                            background: 'var(--bg-tertiary)',
                            border: '1px solid var(--danger, #ef4444)',
                            color: 'var(--danger, #ef4444)',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--space-sm) var(--space-md)',
                          }}
                        >
                          {mediapipeError}
                        </div>
                      )}

                      {mediapipeRun && (
                        <>
                          <div
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1fr 1fr',
                              gap: 'var(--space-sm)',
                            }}
                          >
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {Math.round(
                                  (mediapipeRun.summary.detection_rate || 0) * 100,
                                )}
                                %
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Detection rate
                              </div>
                            </div>
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {mediapipeRun.summary.frames_with_detection}/
                                {mediapipeRun.summary.total_frames}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Frames with detection
                              </div>
                            </div>
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {mediapipeRun.summary.right_hand_selected_count}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Right hand frames
                              </div>
                            </div>
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {mediapipeRun.summary.left_hand_selected_count}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Left hand frames
                              </div>
                            </div>
                          </div>

                          <div
                            style={{
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 'var(--space-xs)',
                              fontSize: '0.75rem',
                              color: 'var(--text-secondary)',
                            }}
                          >
                            <span>
                              <strong>run_id:</strong>{' '}
                              <code
                                style={{
                                  fontFamily: 'var(--font-mono)',
                                  fontSize: '0.7rem',
                                }}
                              >
                                {mediapipeRun.run_id}
                              </code>
                            </span>
                            <span>
                              <strong>fps:</strong> {mediapipeRun.summary.fps.toFixed(2)}{' '}
                              • <strong>frames:</strong>{' '}
                              {mediapipeRun.summary.frame_count} •{' '}
                              <strong>size:</strong> {mediapipeRun.summary.width}×
                              {mediapipeRun.summary.height}
                            </span>
                          </div>

                          <div
                            style={{
                              display: 'flex',
                              flexWrap: 'wrap',
                              gap: 'var(--space-xs)',
                            }}
                          >
                            {mediapipeRun.annotated_video_url && (
                              <a
                                className="btn btn-secondary"
                                style={{ fontSize: '0.75rem', flex: 1 }}
                                href={mediapipeRun.annotated_video_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Open annotated.mp4
                              </a>
                            )}
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={`/storage/mediapipe/runs/${encodeURIComponent(
                                mediapipeRun.run_id,
                              )}/detections.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              detections.json
                            </a>
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={`/storage/mediapipe/runs/${encodeURIComponent(
                                mediapipeRun.run_id,
                              )}/features.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              features.json
                            </a>
                          </div>

                          {mediapipeRun.partial_errors &&
                            mediapipeRun.partial_errors.length > 0 && (
                              <div
                                className="text-small"
                                style={{
                                  color: 'var(--text-muted)',
                                  background: 'var(--bg-tertiary)',
                                  borderRadius: 'var(--radius-md)',
                                  padding: 'var(--space-sm) var(--space-md)',
                                }}
                              >
                                <strong>Warnings:</strong>{' '}
                                {mediapipeRun.partial_errors.join('; ')}
                              </div>
                            )}
                        </>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="sam2">
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-md)',
                        marginTop: 'var(--space-sm)',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 'var(--space-sm)',
                        }}
                      >
                        <Hand size={18} style={{ color: 'var(--accent-primary)' }} />
                        <span className="text-small" style={{ fontWeight: 600 }}>
                          Scissors Tracking (YOLO+SAM2)
                        </span>
                      </div>
                      <p
                        className="text-small"
                        style={{ color: 'var(--text-muted)', margin: 0 }}
                      >
                        Run YOLO to detect the scissors, then use SAM2 to track
                        the scissors trajectory and region across the learner video.
                      </p>

                      <button
                        className="btn btn-primary"
                        style={{
                          width: '100%',
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 'var(--space-xs)',
                        }}
                        disabled={!userVideo || isSam2LearnerProcessing}
                        onClick={runSam2Learner}
                      >
                        {isSam2LearnerProcessing ? (
                          <>
                            <Loader2
                              size={14}
                              style={{ animation: 'spin 1s linear infinite' }}
                            />
                            Running YOLO+SAM2 scissors tracking...
                          </>
                        ) : (
                          <>
                            <Hand size={14} />
                            {sam2LearnerRun ? 'Run YOLO+SAM2 Again' : 'Run YOLO+SAM2'}
                          </>
                        )}
                      </button>

                      {!userVideo && (
                        <p
                          className="text-small"
                          style={{
                            color: 'var(--text-muted)',
                            textAlign: 'center',
                            margin: 0,
                          }}
                        >
                          Upload a practice video to enable YOLO+SAM2 scissors tracking.
                        </p>
                      )}

                      {sam2LearnerError && (
                        <div
                          className="text-small"
                          style={{
                            background: 'var(--bg-tertiary)',
                            border: '1px solid var(--danger, #ef4444)',
                            color: 'var(--danger, #ef4444)',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--space-sm) var(--space-md)',
                          }}
                        >
                          {sam2LearnerError}
                        </div>
                      )}

                      {sam2LearnerRun && (
                        <>
                          <div
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1fr 1fr',
                              gap: 'var(--space-sm)',
                            }}
                          >
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {sam2LearnerRun.processed_frames ?? 0}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Processed frames
                              </div>
                            </div>
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {sam2LearnerRun.successful_masks ?? 0}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Successful masks
                              </div>
                            </div>
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {sam2LearnerRun.device.toUpperCase()}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Device
                              </div>
                            </div>
                            <div
                              className="stat-card"
                              style={{ padding: 'var(--space-sm)' }}
                            >
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.25rem' }}
                              >
                                {sam2LearnerRun.frame_stride}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                frame_stride
                              </div>
                            </div>
                          </div>

                          <div
                            style={{
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 'var(--space-xs)',
                              fontSize: '0.75rem',
                              color: 'var(--text-secondary)',
                            }}
                          >
                            <span>
                              <strong>run_id:</strong>{' '}
                              <code
                                style={{
                                  fontFamily: 'var(--font-mono)',
                                  fontSize: '0.7rem',
                                }}
                              >
                                {sam2LearnerRun.run_id}
                              </code>
                            </span>
                            <span>
                              <strong>main_tracked_feature:</strong>{' '}
                              {sam2LearnerRun.main_tracked_feature || 'bbox_center_trajectory'}
                            </span>
                            <span>
                              <strong>trajectory_stability_score:</strong>{' '}
                              {formatMetricValue(sam2LearnerRun.trajectory_stability_score)} •{' '}
                              <strong>horizontal_drift_range:</strong>{' '}
                              {formatMetricValue(sam2LearnerRun.horizontal_drift_range)} •{' '}
                              <strong>region_stability_score:</strong>{' '}
                              {formatMetricValue(sam2LearnerRun.region_stability_score)}
                            </span>
                            <span>
                              {sam2LearnerRun.expert_reference_available
                                ? 'Expert reference found'
                                : 'Expert reference not found. Run expert preprocessing first.'}
                              {sam2LearnerRun.expert_warning ? ` ${sam2LearnerRun.expert_warning}` : ''}
                            </span>
                            {sam2LearnerRun.tracking_quality_note && (
                              <span>{sam2LearnerRun.tracking_quality_note}</span>
                            )}
                          </div>

                          {sam2LearnerRun.initialization_debug_image_url && (
                            <div style={{ display: 'grid', gap: 4 }}>
                              <span
                                className="text-small"
                                style={{ fontWeight: 600 }}
                              >
                                Initialization prompt (debug)
                              </span>
                              <a
                                href={sam2LearnerRun.initialization_debug_image_url}
                                target="_blank"
                                rel="noreferrer"
                                style={{
                                  display: 'block',
                                  borderRadius: 'var(--radius-sm)',
                                  overflow: 'hidden',
                                  border: '1px solid var(--border-default)',
                                }}
                              >
                                <img
                                  src={sam2LearnerRun.initialization_debug_image_url}
                                  alt="SAM2 initialization prompt overlay"
                                  style={{
                                    display: 'block',
                                    width: '100%',
                                    height: 'auto',
                                  }}
                                />
                              </a>
                              <span
                                className="text-small"
                                style={{
                                  color: 'var(--text-muted)',
                                  fontSize: '0.7rem',
                                }}
                              >
                                YOLO scissors bbox and bbox-center positive point used to initialize SAM2.
                              </span>
                            </div>
                          )}

                          <div
                            style={{
                              display: 'flex',
                              flexWrap: 'wrap',
                              gap: 'var(--space-xs)',
                            }}
                          >
                            {(sam2OverlayBaseUrl || sam2LearnerRun.annotated_video_url) && (
                              <a
                                className="btn btn-secondary"
                                style={{ fontSize: '0.75rem', flex: 1 }}
                                href={sam2OverlayBaseUrl || sam2LearnerRun.annotated_video_url || undefined}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Open sam2_yolo_overlay.mp4
                              </a>
                            )}
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={sam2LearnerRun.raw_json_url || `/storage/outputs/sam2_yolo/runs/${encodeURIComponent(
                                sam2LearnerRun.run_id,
                              )}/raw.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              raw.json
                            </a>
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={sam2LearnerRun.metrics_json_url || `/storage/outputs/sam2_yolo/runs/${encodeURIComponent(
                                sam2LearnerRun.run_id,
                              )}/metrics.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              metrics.json
                            </a>
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={sam2LearnerRun.summary_json_url || `/storage/outputs/sam2_yolo/runs/${encodeURIComponent(
                                sam2LearnerRun.run_id,
                              )}/summary.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              summary.json
                            </a>
                          </div>

                          <div style={{ display: 'grid', gap: 'var(--space-sm)' }}>
                            <div>
                              <span
                                className="text-small"
                                style={{ fontWeight: 600 }}
                              >
                                metrics.json
                              </span>
                              <pre
                                style={{
                                  marginTop: 4,
                                  maxHeight: 140,
                                  overflow: 'auto',
                                  background: 'var(--bg-tertiary)',
                                  borderRadius: 'var(--radius-sm)',
                                  padding: 'var(--space-sm)',
                                  fontSize: '0.7rem',
                                }}
                              >
                                {prettyJson({
                                  trajectory_metrics: sam2LearnerRun.trajectory_metrics,
                                  region_metrics: sam2LearnerRun.region_metrics,
                                  quality_flags: sam2LearnerRun.quality_flags,
                                })}
                              </pre>
                            </div>
                            <div>
                              <span
                                className="text-small"
                                style={{ fontWeight: 600 }}
                              >
                                summary.json
                              </span>
                              <pre
                                style={{
                                  marginTop: 4,
                                  maxHeight: 140,
                                  overflow: 'auto',
                                  background: 'var(--bg-tertiary)',
                                  borderRadius: 'var(--radius-sm)',
                                  padding: 'var(--space-sm)',
                                  fontSize: '0.7rem',
                                }}
                              >
                                {prettyJson({
                                  status: sam2LearnerRun.status,
                                  run_id: sam2LearnerRun.run_id,
                                  device: sam2LearnerRun.device,
                                  frame_stride: sam2LearnerRun.frame_stride,
                                  processed_frames: sam2LearnerRun.processed_frames,
                                  successful_masks: sam2LearnerRun.successful_masks,
                                  failure_frames: sam2LearnerRun.failure_frames,
                                  main_tracked_feature: sam2LearnerRun.main_tracked_feature,
                                  trajectory_stability_score: sam2LearnerRun.trajectory_stability_score,
                                  horizontal_drift_range: sam2LearnerRun.horizontal_drift_range,
                                  region_stability_score: sam2LearnerRun.region_stability_score,
                                  tracking_quality_note: sam2LearnerRun.tracking_quality_note,
                                })}
                              </pre>
                            </div>
                            {Boolean(sam2LearnerRun.raw_preview) && (
                              <div>
                                <div
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    gap: 'var(--space-xs)',
                                  }}
                                >
                                  <span
                                    className="text-small"
                                    style={{ fontWeight: 600 }}
                                  >
                                    raw.json (preview)
                                  </span>
                                  <button
                                    type="button"
                                    className="btn btn-ghost"
                                    style={{ fontSize: '0.7rem' }}
                                    onClick={() =>
                                      setShowSam2RawPreview((prev) => !prev)
                                    }
                                  >
                                    {showSam2RawPreview ? 'Hide' : 'Show'}
                                  </button>
                                </div>
                                {showSam2RawPreview && (
                                  <pre
                                    style={{
                                      marginTop: 4,
                                      maxHeight: 180,
                                      overflow: 'auto',
                                      background: 'var(--bg-tertiary)',
                                      borderRadius: 'var(--radius-sm)',
                                      padding: 'var(--space-sm)',
                                      fontSize: '0.7rem',
                                    }}
                                  >
                                    {prettyJson(sam2LearnerRun.raw_preview)}
                                  </pre>
                                )}
                              </div>
                            )}
                            {Boolean(sam2LearnerRun.tip_tracking_preview) && (
                              <div>
                                <span className="text-small" style={{ fontWeight: 600 }}>
                                  tip_tracking_raw.json (preview)
                                </span>
                                <pre
                                  style={{
                                    marginTop: 4,
                                    maxHeight: 180,
                                    overflow: 'auto',
                                    background: 'var(--bg-tertiary)',
                                    borderRadius: 'var(--radius-sm)',
                                    padding: 'var(--space-sm)',
                                    fontSize: '0.7rem',
                                  }}
                                >
                                  {prettyJson(sam2LearnerRun.tip_tracking_preview)}
                                </pre>
                              </div>
                            )}
                            {Boolean(sam2LearnerRun.manual_init_prompt_preview) && (
                              <div>
                                <span className="text-small" style={{ fontWeight: 600 }}>
                                  manual_init_prompt.json (preview)
                                </span>
                                <pre
                                  style={{
                                    marginTop: 4,
                                    maxHeight: 120,
                                    overflow: 'auto',
                                    background: 'var(--bg-tertiary)',
                                    borderRadius: 'var(--radius-sm)',
                                    padding: 'var(--space-sm)',
                                    fontSize: '0.7rem',
                                  }}
                                >
                                  {prettyJson(sam2LearnerRun.manual_init_prompt_preview)}
                                </pre>
                              </div>
                            )}
                          </div>

                          {sam2LearnerRun.warnings?.length > 0 && (
                            <div
                              className="text-small"
                              style={{
                                color: 'var(--text-muted)',
                                background: 'var(--bg-tertiary)',
                                borderRadius: 'var(--radius-md)',
                                padding: 'var(--space-sm) var(--space-md)',
                              }}
                            >
                              <strong>Warnings:</strong>{' '}
                              {sam2LearnerRun.warnings.join('; ')}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="optical_flow">
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-md)',
                        padding: 'var(--space-sm) 0',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 'var(--space-sm)',
                        }}
                      >
                        <Activity size={18} style={{ color: 'var(--accent-primary)' }} />
                        <span className="text-small" style={{ fontWeight: 600 }}>
                          Motion Instability (Optical Flow)
                        </span>
                      </div>
                      <p
                        className="text-small"
                        style={{ color: 'var(--text-muted)', margin: 0 }}
                      >
                        Run learner-only Optical Flow to estimate vibration and motion
                        stability. These values are side-analysis only and do not affect
                        the evaluation score.
                      </p>

                      <button
                        className="btn btn-primary"
                        style={{
                          width: '100%',
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 'var(--space-xs)',
                        }}
                        disabled={!userVideo || isOpticalFlowProcessing}
                        onClick={runOpticalFlow}
                      >
                        {isOpticalFlowProcessing ? (
                          <>
                            <Loader2
                              size={14}
                              style={{ animation: 'spin 1s linear infinite' }}
                            />
                            Processing...
                          </>
                        ) : (
                          <>
                            <Activity size={14} />
                            {opticalFlowRun ? 'Run Optical Flow Again' : 'Run Optical Flow'}
                          </>
                        )}
                      </button>

                      {!userVideo && (
                        <p
                          className="text-small"
                          style={{
                            color: 'var(--text-muted)',
                            textAlign: 'center',
                            margin: 0,
                          }}
                        >
                          Upload a practice video to enable Optical Flow
                        </p>
                      )}

                      {opticalFlowError && (
                        <div
                          className="text-small"
                          style={{
                            background: 'var(--bg-tertiary)',
                            border: '1px solid var(--danger, #ef4444)',
                            color: 'var(--danger, #ef4444)',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--space-sm) var(--space-md)',
                          }}
                        >
                          {opticalFlowError}
                        </div>
                      )}

                      {opticalFlowRun && (
                        <>
                          <div
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1fr 1fr',
                              gap: 'var(--space-sm)',
                            }}
                          >
                            {[
                              {
                                label: 'Vibration',
                                value: opticalFlowRun.summary.vibration_score,
                              },
                              {
                                label: 'High freq.',
                                value: opticalFlowRun.summary.vibration_high_freq_mean,
                              },
                              {
                                label: 'Stability',
                                value: opticalFlowRun.summary.motion_stability_score,
                              },
                              {
                                label: 'Avg magnitude',
                                value: opticalFlowRun.summary.avg_magnitude,
                              },
                              {
                                label: 'ROI usage',
                                value: opticalFlowRun.summary.roi_usage_ratio,
                              },
                              {
                                label: 'Jitter',
                                value: opticalFlowRun.summary.magnitude_jitter,
                              },
                            ].map((metric) => (
                              <div
                                key={metric.label}
                                className="stat-card"
                                style={{ padding: 'var(--space-sm)' }}
                              >
                                <div
                                  className="stat-value"
                                  style={{ fontSize: '1.25rem' }}
                                >
                                  {formatMetricValue(metric.value)}
                                </div>
                                <div
                                  className="stat-label"
                                  style={{ fontSize: '0.7rem' }}
                                >
                                  {metric.label}
                                </div>
                              </div>
                            ))}
                          </div>

                          <button
                            type="button"
                            className={`btn ${learnerOverlay === 'optical_flow' ? 'btn-primary' : 'btn-secondary'}`}
                            style={{ width: '100%' }}
                            disabled={!opticalFlowVisualizationUrl}
                            onClick={() => {
                              setLearnerOverlay((current) =>
                                current === 'optical_flow' ? 'none' : 'optical_flow',
                              );
                            }}
                          >
                            {learnerOverlay === 'optical_flow'
                              ? 'Show Original Learner Video'
                              : 'Show Optical Flow Visualization'}
                          </button>

                          {opticalFlowVisualizationUrl && (
                            <div
                              className="video-container"
                              style={{
                                aspectRatio: '16/9',
                                border: '1px solid var(--border-default)',
                              }}
                            >
                              <video
                                key={`optical-flow-preview-${opticalFlowLearnerSource ?? opticalFlowVisualizationUrl}`}
                                src={opticalFlowLearnerSource ?? opticalFlowVisualizationUrl}
                                controls
                                muted
                                playsInline
                                preload="metadata"
                              />
                            </div>
                          )}

                          <div
                            style={{
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 'var(--space-xs)',
                              fontSize: '0.75rem',
                              color: 'var(--text-secondary)',
                            }}
                          >
                            <span>
                              <strong>run_id:</strong>{' '}
                              <code
                                style={{
                                  fontFamily: 'var(--font-mono)',
                                  fontSize: '0.7rem',
                                }}
                              >
                                {opticalFlowRun.run_id}
                              </code>
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                  </TabsContent>

                  {/* ── YOLO+Angle+DTW card ───────────────────────────── */}
                  <TabsContent value="yolo_angle">
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-md)',
                        marginTop: 'var(--space-sm)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                        <Triangle size={18} style={{ color: 'var(--accent-primary)' }} />
                        <span className="text-small" style={{ fontWeight: 600 }}>
                          Scissors Angle + DTW Alignment
                        </span>
                      </div>
                      <p className="text-small" style={{ color: 'var(--text-muted)', margin: 0 }}>
                        Run YOLO to detect scissors, extract blade angles frame-by-frame, then
                        align learner and expert angle sequences with Dynamic Time Warping.
                      </p>

                      <button
                        className="btn btn-primary"
                        style={{
                          width: '100%',
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 'var(--space-xs)',
                        }}
                        disabled={!userVideo || isAngleProcessing}
                        onClick={() => void runAngleDtw()}
                      >
                        {isAngleProcessing ? (
                          <>
                            <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                            {angleStatus === 'extracting'
                              ? 'Extracting angles…'
                              : angleStatus === 'dtw'
                                ? 'Running DTW…'
                                : 'Processing…'}
                          </>
                        ) : (
                          <>
                            <Triangle size={14} />
                            {angleDtwResult ? 'Run YOLO+Angle+DTW Again' : 'Run YOLO+Angle+DTW'}
                          </>
                        )}
                      </button>

                      {!userVideo && (
                        <p
                          className="text-small"
                          style={{ color: 'var(--text-muted)', textAlign: 'center', margin: 0 }}
                        >
                          Upload a practice video to enable YOLO+Angle+DTW.
                        </p>
                      )}

                      {angleError && (
                        <div
                          className="text-small"
                          style={{
                            background: 'var(--bg-tertiary)',
                            border: '1px solid var(--danger, #ef4444)',
                            color: 'var(--danger, #ef4444)',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--space-sm) var(--space-md)',
                          }}
                        >
                          {angleError}
                        </div>
                      )}

                      {angleDtwResult && (
                        <>
                          <div
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1fr 1fr',
                              gap: 'var(--space-sm)',
                            }}
                          >
                            {[
                              {
                                label: 'DTW distance',
                                value: angleDtwResult.normalized_dtw_distance,
                              },
                              {
                                label: 'Mean angle diff',
                                value: angleDtwResult.mean_angle_difference,
                              },
                              {
                                label: '✓ OK frames',
                                value: angleDtwResult.ok_frame_count,
                              },
                              {
                                label: '⚠ Medium errors',
                                value: angleDtwResult.medium_error_frame_count,
                              },
                              {
                                label: '✗ High errors',
                                value: angleDtwResult.high_error_frame_count,
                              },
                            ].map((metric) => (
                              <div
                                key={metric.label}
                                className="stat-card"
                                style={{ padding: 'var(--space-sm)' }}
                              >
                                <div className="stat-value" style={{ fontSize: '1.25rem' }}>
                                  {formatMetricValue(metric.value)}
                                </div>
                                <div className="stat-label" style={{ fontSize: '0.7rem' }}>
                                  {metric.label}
                                </div>
                              </div>
                            ))}
                          </div>

                          <button
                            type="button"
                            className={`btn ${learnerOverlay === 'angle' ? 'btn-primary' : 'btn-secondary'}`}
                            style={{ width: '100%' }}
                            onClick={() =>
                              setLearnerOverlay((cur) => cur === 'angle' ? 'none' : 'angle')
                            }
                          >
                            {learnerOverlay === 'angle'
                              ? 'Show Original Learner Video'
                              : 'Show Angle Overlay'}
                          </button>
                        </>
                      )}
                    </div>
                  </TabsContent>
                </Tabs>

              </div>
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

      {/* ─── VLM Coach Feedback ───────────────────────────────────────── */}
      <AnimatePresence>
        {gamePhase === 'result' && (feedbackStatus === 'loading' || feedbackStatus === 'done') && (
          <motion.div
            key="coach-feedback"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            style={{ marginTop: 'var(--space-lg)' }}
          >
            {feedbackStatus === 'loading' && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 'var(--space-sm)',
                  padding: 'var(--space-lg)',
                  color: 'var(--text-muted)',
                  fontSize: '0.875rem',
                }}
              >
                <Loader2 size={16} style={{ animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                Your Crafting Coach is reviewing your practice...
              </div>
            )}
            {feedbackStatus === 'done' && feedbackText && (
              <div
                style={{
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-lg)',
                  padding: 'var(--space-xl)',
                }}
              >
                <p
                  style={{
                    fontWeight: 700,
                    fontSize: '1.05rem',
                    color: 'var(--accent-primary)',
                    margin: '0 0 var(--space-md) 0',
                  }}
                >
                  🎓 Your Crafting Coach
                </p>
                <p
                  style={{
                    lineHeight: 1.75,
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    color: 'var(--text-primary)',
                    fontSize: '0.9375rem',
                  }}
                >
                  {renderFeedback(feedbackText)}
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

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
