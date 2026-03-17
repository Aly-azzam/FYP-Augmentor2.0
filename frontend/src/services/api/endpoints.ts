import type { EvaluationResult } from '../../types';
import { generateMockEvaluation } from '../mock/evaluation';

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function uploadPracticeVideo(
  file: File,
  onProgress?: (progress: number) => void,
): Promise<{ videoId: string; url: string }> {
  const totalChunks = 10;
  for (let i = 1; i <= totalChunks; i++) {
    await delay(200);
    onProgress?.(Math.round((i / totalChunks) * 100));
  }
  return {
    videoId: `video-${Date.now()}`,
    url: URL.createObjectURL(file),
  };
}

export async function validateDomain(
  videoId: string,
): Promise<{ isValid: boolean; confidence: number; domain: string }> {
  await delay(1500);
  return {
    isValid: true,
    confidence: 0.94,
    domain: 'crafting',
  };
}

interface PipelineStage {
  name: string;
  duration: number;
}

const pipelineStages: PipelineStage[] = [
  { name: 'Preprocessing video frames', duration: 800 },
  { name: 'Extracting pose keypoints', duration: 1200 },
  { name: 'Analyzing motion trajectories', duration: 1000 },
  { name: 'Computing metric deviations', duration: 900 },
  { name: 'Generating evaluation report', duration: 600 },
];

export async function runEvaluation(
  courseId: string,
  clipId: string,
  _videoId: string,
  onStage?: (stage: string, progress: number) => void,
): Promise<EvaluationResult> {
  const totalDuration = pipelineStages.reduce((sum, s) => sum + s.duration, 0);
  let elapsed = 0;

  for (const stage of pipelineStages) {
    onStage?.(stage.name, Math.round((elapsed / totalDuration) * 100));
    await delay(stage.duration);
    elapsed += stage.duration;
  }

  onStage?.('Complete', 100);
  return generateMockEvaluation(courseId, clipId);
}

export async function getExplanation(
  evaluationId: string,
): Promise<{ explanation: string; suggestions: string[] }> {
  await delay(1000);
  return {
    explanation: `Detailed analysis for evaluation ${evaluationId}: Your technique shows consistent improvement in trajectory control. The primary area for development is maintaining tool angle consistency during transitions between movements.`,
    suggestions: [
      'Focus on keeping your elbow anchored during the pulling phase',
      'Reduce speed during direction changes for smoother trajectories',
      'Practice the transition movement separately before combining with full technique',
      'Review the instructor video at 0.5x speed for the highlighted segments',
    ],
  };
}
