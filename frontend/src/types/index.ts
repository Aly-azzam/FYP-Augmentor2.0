export interface Course {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  totalClips: number;
  estimatedTime: string;
  progress: number;
  thumbnail: string;
  category: string;
  instructor: string;
}

export interface VideoClip {
  id: string;
  title: string;
  duration: number;
  description: string;
  thumbnail: string;
  keyPoints: string[];
  expertVideoUrl?: string;
}

export interface MetricSet {
  angleDeviation: number;
  trajectoryDeviation: number;
  velocityDifference: number;
  toolAlignmentDeviation: number;
}

export interface EvaluationResult {
  id: string;
  courseId: string;
  clipId: string;
  date: string;
  score: number;
  label: string;
  metrics: MetricSet;
  explanation: string;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlockedAt: string | null;
  progress: number;
  maxProgress: number;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

export interface UserProfile {
  name: string;
  email: string;
  joinedAt: string;
  totalEvaluations: number;
  averageScore: number;
  streakDays: number;
  level: number;
  xp: number;
  nextLevelXp: number;
}

export interface RobotTip {
  id: string;
  message: string;
  context: string;
  priority: 'high' | 'medium' | 'low';
}

export interface ProgressDataPoint {
  date: string;
  score: number;
  metric1: number;
  metric2: number;
  metric3: number;
  metric4: number;
}

export interface Timer {
  id: string;
  name: string;
  elapsed: number;
  isRunning: boolean;
  timestamps: number[];
}

export type DrawingTool =
  | 'select'
  | 'arrow'
  | 'line'
  | 'rectangle'
  | 'circle'
  | 'pen'
  | 'angle'
  | 'calibrate'
  | 'track';
