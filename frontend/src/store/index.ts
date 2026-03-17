import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { DrawingTool, EvaluationResult, Timer } from '../types';

// ─── Theme Store ───────────────────────────────────────────────────────────────

interface ThemeState {
  isDark: boolean;
  toggleTheme: () => void;
  setTheme: (isDark: boolean) => void;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      isDark: true,
      toggleTheme: () => set((s) => ({ isDark: !s.isDark })),
      setTheme: (isDark) => set({ isDark }),
    }),
    { name: 'augmentor-theme' },
  ),
);

// ─── Video Sync Store ──────────────────────────────────────────────────────────

interface VideoSyncState {
  isSynced: boolean;
  playbackRate: number;
  isPlaying: boolean;
  currentTime: number;
  offset: number;
  setIsSynced: (v: boolean) => void;
  setPlaybackRate: (v: number) => void;
  setIsPlaying: (v: boolean) => void;
  setCurrentTime: (v: number) => void;
  setOffset: (v: number) => void;
}

export const useVideoSyncStore = create<VideoSyncState>((set) => ({
  isSynced: true,
  playbackRate: 1,
  isPlaying: false,
  currentTime: 0,
  offset: 0,
  setIsSynced: (isSynced) => set({ isSynced }),
  setPlaybackRate: (playbackRate) => set({ playbackRate }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setCurrentTime: (currentTime) => set({ currentTime }),
  setOffset: (offset) => set({ offset }),
}));

// ─── Drawing Store ─────────────────────────────────────────────────────────────

interface TrackedPoint {
  x: number;
  y: number;
  frame: number;
}

interface Measurement {
  id: string;
  type: string;
  value: number;
  unit: string;
}

interface DrawingState {
  activeTool: DrawingTool;
  toolColor: string;
  toolThickness: number;
  showAllFrames: boolean;
  trackedPoints: TrackedPoint[];
  measurements: Measurement[];
  setActiveTool: (tool: DrawingTool) => void;
  setToolColor: (color: string) => void;
  setToolThickness: (thickness: number) => void;
  setShowAllFrames: (show: boolean) => void;
  addTrackedPoint: (point: TrackedPoint) => void;
  clearTrackedPoints: () => void;
  addMeasurement: (m: Measurement) => void;
  clearMeasurements: () => void;
}

export const useDrawingStore = create<DrawingState>((set) => ({
  activeTool: 'select',
  toolColor: '#2563EB',
  toolThickness: 2,
  showAllFrames: false,
  trackedPoints: [],
  measurements: [],
  setActiveTool: (activeTool) => set({ activeTool }),
  setToolColor: (toolColor) => set({ toolColor }),
  setToolThickness: (toolThickness) => set({ toolThickness }),
  setShowAllFrames: (showAllFrames) => set({ showAllFrames }),
  addTrackedPoint: (point) =>
    set((s) => ({ trackedPoints: [...s.trackedPoints, point] })),
  clearTrackedPoints: () => set({ trackedPoints: [] }),
  addMeasurement: (m) =>
    set((s) => ({ measurements: [...s.measurements, m] })),
  clearMeasurements: () => set({ measurements: [] }),
}));

// ─── Evaluation Store ──────────────────────────────────────────────────────────

interface EvaluationState {
  isEvaluating: boolean;
  evaluationStep: number;
  evaluationProgress: number;
  evaluationResult: EvaluationResult | null;
  startEvaluation: () => void;
  setEvaluationStep: (step: number) => void;
  setEvaluationProgress: (progress: number) => void;
  setEvaluationResult: (result: EvaluationResult | null) => void;
  resetEvaluation: () => void;
}

export const useEvaluationStore = create<EvaluationState>((set) => ({
  isEvaluating: false,
  evaluationStep: 0,
  evaluationProgress: 0,
  evaluationResult: null,
  startEvaluation: () =>
    set({ isEvaluating: true, evaluationStep: 0, evaluationProgress: 0 }),
  setEvaluationStep: (evaluationStep) => set({ evaluationStep }),
  setEvaluationProgress: (evaluationProgress) => set({ evaluationProgress }),
  setEvaluationResult: (evaluationResult) =>
    set({ evaluationResult, isEvaluating: false }),
  resetEvaluation: () =>
    set({
      isEvaluating: false,
      evaluationStep: 0,
      evaluationProgress: 0,
      evaluationResult: null,
    }),
}));

// ─── UI Store ──────────────────────────────────────────────────────────────────

interface UIState {
  showRobot: boolean;
  robotMessage: string | null;
  showGuidedTour: boolean;
  tourStep: number;
  activeModal: string | null;
  sidebarOpen: boolean;
  setShowRobot: (v: boolean) => void;
  setRobotMessage: (msg: string | null) => void;
  setShowGuidedTour: (v: boolean) => void;
  setTourStep: (step: number) => void;
  setActiveModal: (modal: string | null) => void;
  setSidebarOpen: (open: boolean) => void;
  nextTourStep: () => void;
  prevTourStep: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  showRobot: true,
  robotMessage: null,
  showGuidedTour: false,
  tourStep: 0,
  activeModal: null,
  sidebarOpen: false,
  setShowRobot: (showRobot) => set({ showRobot }),
  setRobotMessage: (robotMessage) => set({ robotMessage }),
  setShowGuidedTour: (showGuidedTour) => set({ showGuidedTour }),
  setTourStep: (tourStep) => set({ tourStep }),
  setActiveModal: (activeModal) => set({ activeModal }),
  setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
  nextTourStep: () => set((s) => ({ tourStep: s.tourStep + 1 })),
  prevTourStep: () => set((s) => ({ tourStep: Math.max(0, s.tourStep - 1) })),
}));

// ─── Course Store ──────────────────────────────────────────────────────────────

interface CourseState {
  selectedCourse: string | null;
  selectedClip: string | null;
  userVideo: File | null;
  setSelectedCourse: (id: string | null) => void;
  setSelectedClip: (id: string | null) => void;
  setUserVideo: (file: File | null) => void;
}

export const useCourseStore = create<CourseState>((set) => ({
  selectedCourse: null,
  selectedClip: null,
  userVideo: null,
  setSelectedCourse: (selectedCourse) => set({ selectedCourse }),
  setSelectedClip: (selectedClip) => set({ selectedClip }),
  setUserVideo: (userVideo) => set({ userVideo }),
}));

// ─── Timer Store ───────────────────────────────────────────────────────────────

interface TimerState {
  timers: Timer[];
  activeTimer: string | null;
  addTimer: (timer: Timer) => void;
  removeTimer: (id: string) => void;
  startTimer: (id: string) => void;
  stopTimer: (id: string) => void;
  resetTimer: (id: string) => void;
  addTimestamp: (id: string, timestamp: number) => void;
  updateElapsed: (id: string, elapsed: number) => void;
}

export const useTimerStore = create<TimerState>((set) => ({
  timers: [
    { id: 'timer-1', name: 'Practice Timer', elapsed: 0, isRunning: false, timestamps: [] },
    { id: 'timer-2', name: 'Session Timer', elapsed: 0, isRunning: false, timestamps: [] },
    { id: 'timer-3', name: 'Break Timer', elapsed: 0, isRunning: false, timestamps: [] },
  ],
  activeTimer: null,
  addTimer: (timer) => set((s) => ({ timers: [...s.timers, timer] })),
  removeTimer: (id) =>
    set((s) => ({ timers: s.timers.filter((t) => t.id !== id) })),
  startTimer: (id) =>
    set((s) => ({
      activeTimer: id,
      timers: s.timers.map((t) =>
        t.id === id ? { ...t, isRunning: true } : t,
      ),
    })),
  stopTimer: (id) =>
    set((s) => ({
      timers: s.timers.map((t) =>
        t.id === id ? { ...t, isRunning: false } : t,
      ),
    })),
  resetTimer: (id) =>
    set((s) => ({
      timers: s.timers.map((t) =>
        t.id === id ? { ...t, elapsed: 0, isRunning: false, timestamps: [] } : t,
      ),
    })),
  addTimestamp: (id, timestamp) =>
    set((s) => ({
      timers: s.timers.map((t) =>
        t.id === id ? { ...t, timestamps: [...t.timestamps, timestamp] } : t,
      ),
    })),
  updateElapsed: (id, elapsed) =>
    set((s) => ({
      timers: s.timers.map((t) =>
        t.id === id ? { ...t, elapsed } : t,
      ),
    })),
}));
