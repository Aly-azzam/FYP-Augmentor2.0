import type { EvaluationResult } from '../../types';

export const evaluationHistory: EvaluationResult[] = [
  {
    id: 'eval-1',
    courseId: 'vase-making-101',
    clipId: 'vase-making-101-clip-3',
    date: '2025-01-15T10:30:00Z',
    score: 92,
    label: 'Excellent',
    metrics: {
      angleDeviation: 2.1,
      trajectoryDeviation: 1.8,
      velocityDifference: 3.2,
      toolAlignmentDeviation: 1.5,
    },
    explanation:
      'Outstanding performance. Your hand positioning and wall pulling technique closely matches the instructor. Minor velocity variation detected during the narrowing phase.',
  },
  {
    id: 'eval-2',
    courseId: 'vase-making-101',
    clipId: 'vase-making-101-clip-5',
    date: '2025-01-14T14:15:00Z',
    score: 78,
    label: 'Good',
    metrics: {
      angleDeviation: 5.3,
      trajectoryDeviation: 4.1,
      velocityDifference: 6.7,
      toolAlignmentDeviation: 3.8,
    },
    explanation:
      'Good technique overall. Focus on maintaining consistent pressure during the neck narrowing. Your trajectory shows slight drift to the left side.',
  },
  {
    id: 'eval-3',
    courseId: 'wood-carving-basics',
    clipId: 'wood-carving-basics-clip-1',
    date: '2025-01-13T09:00:00Z',
    score: 85,
    label: 'Very Good',
    metrics: {
      angleDeviation: 3.5,
      trajectoryDeviation: 2.9,
      velocityDifference: 4.1,
      toolAlignmentDeviation: 2.7,
    },
    explanation:
      'Very good carving technique. Your grain selection approach is solid. Work on keeping the gouge angle more consistent during long cuts.',
  },
  {
    id: 'eval-4',
    courseId: 'vase-making-101',
    clipId: 'vase-making-101-clip-1',
    date: '2025-01-12T16:45:00Z',
    score: 61,
    label: 'Fair',
    metrics: {
      angleDeviation: 8.2,
      trajectoryDeviation: 7.5,
      velocityDifference: 9.3,
      toolAlignmentDeviation: 6.1,
    },
    explanation:
      'Fair attempt at centering. The clay was off-center during the initial phase. Focus on applying even pressure with both hands and keeping your elbows anchored.',
  },
  {
    id: 'eval-5',
    courseId: 'vase-making-101',
    clipId: 'vase-making-101-clip-7',
    date: '2025-01-11T11:20:00Z',
    score: 45,
    label: 'Needs Improvement',
    metrics: {
      angleDeviation: 12.4,
      trajectoryDeviation: 10.8,
      velocityDifference: 14.2,
      toolAlignmentDeviation: 9.6,
    },
    explanation:
      'The base formation needs significant practice. Review the instructor video focusing on hand placement and the amount of water used. Try slowing down your movements.',
  },
];

export function generateMockEvaluation(
  courseId: string,
  clipId: string,
): EvaluationResult {
  const score = Math.floor(Math.random() * 50) + 50;
  const label =
    score >= 90
      ? 'Excellent'
      : score >= 80
        ? 'Very Good'
        : score >= 70
          ? 'Good'
          : score >= 50
            ? 'Fair'
            : 'Needs Improvement';

  const base = (100 - score) / 5;

  return {
    id: `eval-${Date.now()}`,
    courseId,
    clipId,
    date: new Date().toISOString(),
    score,
    label,
    metrics: {
      angleDeviation: parseFloat((base + Math.random() * 3).toFixed(1)),
      trajectoryDeviation: parseFloat((base + Math.random() * 2).toFixed(1)),
      velocityDifference: parseFloat((base + Math.random() * 4).toFixed(1)),
      toolAlignmentDeviation: parseFloat((base + Math.random() * 2.5).toFixed(1)),
    },
    explanation: `Your performance scored ${score}/100. ${
      score >= 80
        ? 'Great work! Keep refining your technique for even better results.'
        : score >= 60
          ? 'Good progress. Focus on the areas highlighted in the metrics for improvement.'
          : 'Keep practicing! Review the instructor video and pay attention to the key points.'
    }`,
  };
}

export function scoreColor(score: number): string {
  if (score >= 90) return '#22c55e';
  if (score >= 70) return '#3b82f6';
  if (score >= 50) return '#eab308';
  return '#ef4444';
}
