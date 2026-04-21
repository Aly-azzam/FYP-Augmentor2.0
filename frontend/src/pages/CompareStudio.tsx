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
  Scissors,
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

// ── MediaPipe integration ──────────────────────────────────────────────────

interface MediaPipeRunSummary {
  run_id: string;
  source_video_path: string;
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

type InspectionModel = 'mediapipe' | 'sam2' | 'optical_flow';

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
  run_folder: string;
  raw_json_path: string;
  summary_json_path: string;
  metadata_json_path: string;
  annotated_video_path: string | null;
  annotated_video_url: string | null;
  source_video_path: string;
  initialization_debug_image_path: string | null;
  initialization_debug_image_url: string | null;
  device: string;
  frame_stride: number;
  metadata: Sam2LearnerMetadata;
  summary: Sam2LearnerSummary;
  raw_preview: unknown | null;
  mediapipe?: Sam2LearnerMediaPipeInfo | null;
  warnings: string[];
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

  // MediaPipe integration state.
  const [mediapipeRun, setMediapipeRun] = useState<MediaPipeRunResult | null>(null);
  const [isMediapipeProcessing, setIsMediapipeProcessing] = useState(false);
  const [mediapipeError, setMediapipeError] = useState<string | null>(null);
  const [mediapipeVideoVersion, setMediapipeVideoVersion] = useState(0);
  const [selectedExpertVideoId, setSelectedExpertVideoId] = useState<string | null>(null);
  const [inspectionModel, setInspectionModel] = useState<InspectionModel>('mediapipe');
  // Learner overlay: show the raw uploaded video, the MediaPipe annotated
  // output, or the SAM 2 annotated output (Compare Studio is learner-focused).
  const [learnerOverlay, setLearnerOverlay] = useState<'none' | 'mediapipe' | 'sam2'>('none');

  // SAM2 learner (Compare Studio) integration state. This is the
  // learner-focused SAM2 flow: upload -> MediaPipe -> SAM2 -> annotated.
  const [sam2LearnerRun, setSam2LearnerRun] = useState<Sam2LearnerResult | null>(null);
  const [isSam2LearnerProcessing, setIsSam2LearnerProcessing] = useState(false);
  const [sam2LearnerError, setSam2LearnerError] = useState<string | null>(null);
  const [sam2LearnerVideoVersion, setSam2LearnerVideoVersion] = useState(0);
  const [showSam2RawPreview, setShowSam2RawPreview] = useState(false);

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
  const hasSelectedExpertReference = Boolean(selectedClip || requestedClipId);

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
      setSelectedExpertVideoId(null);
      setExpertCurrentTime(0);
      setExpertDuration(0);
    }
  }, [clip]);

  useEffect(() => {
    let isCancelled = false;

    const loadSelectedExpertVideo = async () => {
      try {
        // 1) If selectedClip is a real chapter UUID, resolve chapter expert video first.
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

        // 3) Last-resort fallback for legacy mock clips.
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
          setExpertVideoError('No expert video is linked yet. Upload one from Expert Upload.');
        }
      } catch {
        if (!isCancelled) {
          setExpertVideoUrl(null);
          setSelectedExpertVideoId(null);
          setExpertVideoError('No expert video is linked yet. Upload one from Expert Upload.');
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
        setLearnerOverlay('none');
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

  // ── SAM 2 learner run (Compare Studio pipeline) ───────────────────────────
  //
  // Triggered by the "Run SAM2" button. The backend endpoint runs
  // MediaPipe first and then SAM 2 on the same learner video, so a
  // single click gives us the complete pipeline output (annotated video,
  // metadata/summary, and the initialization debug image).
  const runSam2Learner = useCallback(async () => {
    if (!userVideo) {
      toast.error('Upload a practice video first');
      return;
    }

    setIsSam2LearnerProcessing(true);
    setSam2LearnerError(null);

    try {
      const formData = new FormData();
      formData.append('file', userVideo);

      const response = await fetch('/api/sam2/process-upload', {
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
        let message: string;
        if (detail && typeof detail === 'object') {
          if (detail.error_type === 'SAM2AssetsMissingError') {
            message =
              'SAM2 model assets are missing. Backend SAM2 processing cannot run yet.';
          } else if (detail.stage === 'mediapipe') {
            message = `MediaPipe step failed: ${detail.message ?? 'unknown error'}`;
          } else if (typeof detail.message === 'string') {
            message = detail.message;
          } else {
            message = `SAM2 failed with status ${response.status}`;
          }
        } else if (typeof detail === 'string') {
          message = detail;
        } else {
          message = `SAM2 failed with status ${response.status}`;
        }
        throw new Error(message);
      }

      const result = payload as Sam2LearnerResult;
      setSam2LearnerRun(result);
      setSam2LearnerVideoVersion(Date.now());
      if (result.mediapipe?.annotated_video_url) {
        setMediapipeRun({
          run_id: result.mediapipe.run_id,
          run_folder: result.mediapipe.run_folder,
          detections_json_path: '',
          features_json_path: result.mediapipe.features_json_path,
          metadata_json_path: result.mediapipe.metadata_json_path,
          annotated_video_path: result.mediapipe.annotated_video_path,
          annotated_video_url: result.mediapipe.annotated_video_url,
          summary: {
            run_id: result.mediapipe.run_id,
            source_video_path: result.source_video_path,
            fps: result.metadata.fps,
            frame_count: result.metadata.frame_count,
            width: result.metadata.width,
            height: result.metadata.height,
            created_at: result.metadata.created_at,
            selected_hand_policy: 'prefer_right_else_first',
            total_frames: result.mediapipe.total_frames,
            frames_with_detection: result.mediapipe.frames_with_detection,
            detection_rate: result.mediapipe.detection_rate,
            right_hand_selected_count: 0,
            left_hand_selected_count: 0,
          },
        });
        setMediapipeVideoVersion(Date.now());
      }
      if (result.annotated_video_url) {
        setLearnerOverlay('sam2');
      }

      const maskedPct = Math.round((result.summary.detection_rate || 0) * 100);
      toast.success(
        `SAM2 ready — ${result.device.toUpperCase()} • stride ${result.frame_stride} • ${maskedPct}% frames with mask`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'SAM2 pipeline failed.';
      setSam2LearnerError(message);
      toast.error(message);
    } finally {
      setIsSam2LearnerProcessing(false);
    }
  }, [userVideo]);

  // Compare Studio is learner-focused. The SAM 2 tab now shows the
  // learner pipeline (MediaPipe -> SAM 2) run on the uploaded practice
  // video; we intentionally do NOT surface the expert SAM 2 inspection
  // payload here anymore.

  const mediapipeAnnotatedSource =
    mediapipeRun?.annotated_video_url && mediapipeVideoVersion > 0
      ? `${mediapipeRun.annotated_video_url}${mediapipeRun.annotated_video_url.includes('?') ? '&' : '?'}v=${mediapipeVideoVersion}`
      : mediapipeRun?.annotated_video_url ?? null;

  const sam2AnnotatedSource =
    sam2LearnerRun?.annotated_video_url && sam2LearnerVideoVersion > 0
      ? `${sam2LearnerRun.annotated_video_url}${sam2LearnerRun.annotated_video_url.includes('?') ? '&' : '?'}v=${sam2LearnerVideoVersion}`
      : sam2LearnerRun?.annotated_video_url ?? null;

  const learnerVideoSource = (() => {
    if (learnerOverlay === 'mediapipe' && mediapipeAnnotatedSource) {
      return mediapipeAnnotatedSource;
    }
    if (learnerOverlay === 'sam2' && sam2AnnotatedSource) {
      return sam2AnnotatedSource;
    }
    return userVideoUrl;
  })();

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
                  key={expertVideoUrl ?? 'missing-expert-video'}
                  src={expertVideoUrl ?? undefined}
                  poster={clip?.thumbnail}
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
                {(mediapipeRun?.annotated_video_url || sam2LearnerRun?.annotated_video_url) && (
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
                    {sam2LearnerRun?.annotated_video_url && (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={learnerOverlay === 'sam2'}
                        className={`btn ${learnerOverlay === 'sam2' ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ borderRadius: 0, fontSize: '0.75rem', padding: 'var(--space-xs) var(--space-sm)' }}
                        onClick={() => setLearnerOverlay('sam2')}
                      >
                        <Scissors size={12} style={{ marginRight: 4 }} />
                        SAM2
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
                  onTimeUpdate={() => {
                    if (learnerVideoRef.current)
                      setLearnerCurrentTime(learnerVideoRef.current.currentTime);
                  }}
                  onLoadedMetadata={() => {
                    if (learnerVideoRef.current)
                      setLearnerDuration(learnerVideoRef.current.duration);
                  }}
                />
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
                {learnerOverlay === 'sam2' && sam2LearnerRun?.annotated_video_url && (
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
                    <Scissors size={10} />
                    SAM2
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
          <Tabs defaultValue="tools">
            <TabsList>
              <TabsTrigger value="tools">Tools</TabsTrigger>
              <TabsTrigger value="evaluate">Evaluate</TabsTrigger>
              <TabsTrigger value="mediapipe">MediaPipe</TabsTrigger>
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
                    <TabsTrigger value="sam2">SAM2</TabsTrigger>
                    <TabsTrigger value="optical_flow">Optical Flow</TabsTrigger>
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
                        <Scissors size={18} style={{ color: 'var(--accent-primary)' }} />
                        <span className="text-small" style={{ fontWeight: 600 }}>
                          Hand Segmentation (SAM2)
                        </span>
                      </div>
                      <p
                        className="text-small"
                        style={{ color: 'var(--text-muted)', margin: 0 }}
                      >
                        Run the MediaPipe -&gt; SAM2 pipeline on your practice
                        video. MediaPipe builds the initialization prompt
                        automatically; SAM2 then segments the hand and writes
                        the annotated video, masks and summary. Toggle
                        between Original / MediaPipe / SAM2 above the learner
                        video to compare outputs.
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
                            Processing (MediaPipe -&gt; SAM2)...
                          </>
                        ) : (
                          <>
                            <Scissors size={14} />
                            {sam2LearnerRun ? 'Run Again' : 'Run SAM2'}
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
                          Upload a practice video to enable SAM2.
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
                                {Math.round(
                                  (sam2LearnerRun.summary.detection_rate || 0) * 100,
                                )}
                                %
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Frames with mask
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
                                {sam2LearnerRun.summary.frames_with_mask}/
                                {sam2LearnerRun.summary.total_frames}
                              </div>
                              <div
                                className="stat-label"
                                style={{ fontSize: '0.7rem' }}
                              >
                                Masked / total
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
                                {sam2LearnerRun.metadata.gpu_name
                                  ? ` (${sam2LearnerRun.metadata.gpu_name})`
                                  : ''}
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
                              <strong>fps:</strong>{' '}
                              {sam2LearnerRun.metadata.fps.toFixed(2)} •{' '}
                              <strong>frames:</strong>{' '}
                              {sam2LearnerRun.metadata.frame_count} •{' '}
                              <strong>size:</strong>{' '}
                              {sam2LearnerRun.metadata.width}×
                              {sam2LearnerRun.metadata.height}
                            </span>
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
                                Point + bounding box used to initialize SAM2
                                on the first valid MediaPipe frame.
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
                            {sam2LearnerRun.annotated_video_url && (
                              <a
                                className="btn btn-secondary"
                                style={{ fontSize: '0.75rem', flex: 1 }}
                                href={sam2LearnerRun.annotated_video_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Open annotated.mp4
                              </a>
                            )}
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={`/storage/sam2/runs/${encodeURIComponent(
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
                              href={`/storage/sam2/runs/${encodeURIComponent(
                                sam2LearnerRun.run_id,
                              )}/summary.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              summary.json
                            </a>
                            <a
                              className="btn btn-ghost"
                              style={{ fontSize: '0.75rem', flex: 1 }}
                              href={`/storage/sam2/runs/${encodeURIComponent(
                                sam2LearnerRun.run_id,
                              )}/metadata.json`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              metadata.json
                            </a>
                          </div>

                          <div style={{ display: 'grid', gap: 'var(--space-sm)' }}>
                            <div>
                              <span
                                className="text-small"
                                style={{ fontWeight: 600 }}
                              >
                                metadata.json
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
                                {prettyJson(sam2LearnerRun.metadata)}
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
                                {prettyJson(sam2LearnerRun.summary)}
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
                      className="text-small"
                      style={{
                        color: 'var(--text-muted)',
                        marginTop: 'var(--space-sm)',
                        background: 'var(--bg-tertiary)',
                        borderRadius: 'var(--radius-md)',
                        padding: 'var(--space-sm) var(--space-md)',
                      }}
                    >
                      Optical Flow inspection is reserved for later and will appear here in the same panel.
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
