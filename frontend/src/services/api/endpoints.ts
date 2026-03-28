import { getEvaluationResult, startEvaluation } from '../../api/evaluationApi';

export async function uploadPracticeVideo(
  file: File,
  onProgress?: (progress: number) => void,
): Promise<{ videoId: string; url: string }> {
  onProgress?.(100);
  return {
    videoId: `video-${Date.now()}`,
    url: URL.createObjectURL(file),
  };
}

export async function validateDomain(
  _videoId: string,
): Promise<{ isValid: boolean; confidence: number; domain: string }> {
  return {
    isValid: true,
    confidence: 1,
    domain: 'unknown',
  };
}

export async function runEvaluation(
  _courseId: string,
  _clipId: string,
  videoId: string,
  onStage?: (stage: string, progress: number) => void,
): Promise<any> {
  onStage?.('Starting evaluation', 10);
  const evaluationId = await startEvaluation({ attempt_id: videoId });
  onStage?.('Fetching result', 80);
  const result = await getEvaluationResult(evaluationId);
  onStage?.('Complete', 100);
  return result;
}

export async function getExplanation(
  evaluationId: string,
): Promise<{ explanation: string; suggestions: string[] }> {
  const result = await getEvaluationResult(evaluationId);
  return {
    explanation: result?.explanation?.explanation || '',
    suggestions: [],
  };
}
