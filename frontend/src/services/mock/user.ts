import type { UserProfile, RobotTip, ProgressDataPoint } from '../../types';

export const userProfile: UserProfile = {
  name: 'Alex Rivera',
  email: 'alex.rivera@example.com',
  joinedAt: '2024-12-01T00:00:00Z',
  totalEvaluations: 47,
  averageScore: 76,
  streakDays: 5,
  level: 12,
  xp: 2840,
  nextLevelXp: 3000,
};

export const robotTips: RobotTip[] = [
  {
    id: 'tip-1',
    message:
      'Welcome to AugMentor! I\'m here to help you improve your craft skills. Start by selecting a course from the library.',
    context: 'dashboard',
    priority: 'high',
  },
  {
    id: 'tip-2',
    message:
      'Try recording your practice session and uploading it for evaluation. I\'ll compare your technique with the instructor\'s.',
    context: 'compare',
    priority: 'high',
  },
  {
    id: 'tip-3',
    message:
      'Use the drawing tools to annotate specific points on the video. This helps track your movement patterns over time.',
    context: 'compare',
    priority: 'medium',
  },
  {
    id: 'tip-4',
    message:
      'Your angle deviation has been improving! Keep focusing on maintaining consistent hand positioning.',
    context: 'progress',
    priority: 'medium',
  },
  {
    id: 'tip-5',
    message:
      'Don\'t forget to take breaks during practice. Rest is important for muscle memory development.',
    context: 'general',
    priority: 'low',
  },
  {
    id: 'tip-6',
    message:
      'Check out the achievements page to see your progress milestones. You\'re close to unlocking "Week Warrior"!',
    context: 'achievements',
    priority: 'medium',
  },
  {
    id: 'tip-7',
    message:
      'Slow down your movements when practicing new techniques. Speed comes with mastery.',
    context: 'compare',
    priority: 'high',
  },
  {
    id: 'tip-8',
    message:
      'Review your evaluation history to identify patterns and areas where you\'ve improved the most.',
    context: 'history',
    priority: 'low',
  },
];

export const progressData: ProgressDataPoint[] = [
  { date: '2025-01-09', score: 52, metric1: 10.5, metric2: 9.2, metric3: 12.1, metric4: 8.4 },
  { date: '2025-01-10', score: 58, metric1: 9.1, metric2: 8.5, metric3: 11.0, metric4: 7.6 },
  { date: '2025-01-11', score: 45, metric1: 12.4, metric2: 10.8, metric3: 14.2, metric4: 9.6 },
  { date: '2025-01-12', score: 61, metric1: 8.2, metric2: 7.5, metric3: 9.3, metric4: 6.1 },
  { date: '2025-01-13', score: 85, metric1: 3.5, metric2: 2.9, metric3: 4.1, metric4: 2.7 },
  { date: '2025-01-14', score: 78, metric1: 5.3, metric2: 4.1, metric3: 6.7, metric4: 3.8 },
  { date: '2025-01-15', score: 92, metric1: 2.1, metric2: 1.8, metric3: 3.2, metric4: 1.5 },
];
