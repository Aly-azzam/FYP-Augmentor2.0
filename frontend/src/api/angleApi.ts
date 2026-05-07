// Typed fetch wrappers for the YOLO + Angle + DTW backend pipeline.

export interface AngleDtwSummary {
  dtw_distance: number | null;
  normalized_dtw_distance: number | null;
  mean_angle_difference: number | null;
  high_error_frame_count: number | null;
  medium_error_frame_count: number | null;
  ok_frame_count: number | null;
}

export interface RunLearnerAngleResponse {
  status: 'done';
  run_id: string;
  dtw_path: string;
  learner_angles_path: string;
  summary: AngleDtwSummary;
}

export interface ExpertAngleStatusResponse {
  exists: boolean;
  expert_name: string;
}

export interface GeneratePreviewResponse {
  status: 'done';
  run_id: string;
  preview_path: string;
}

/** Run the full learner angle + DTW pipeline.
 *  Sends the video file as multipart form data.
 *  Returns a unique run_id to reference all subsequent calls. */
export async function runLearnerAngle(
  file: File,
  expertName: string,
): Promise<RunLearnerAngleResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('expert_name', expertName);

  const res = await fetch('/api/angle/run-learner', {
    method: 'POST',
    body: form,
  });

  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail = (payload as any)?.detail;
    throw new Error(
      (typeof detail === 'string' ? detail : null) ??
        `Angle pipeline failed with status ${res.status}`,
    );
  }

  return payload as RunLearnerAngleResponse;
}

/** Fetch the saved dtw_alignment.json for a given run. */
export async function getLearnerAngleResult(runId: string): Promise<unknown> {
  const res = await fetch(
    `/api/angle/learner/${encodeURIComponent(runId)}/result`,
  );

  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail = (payload as any)?.detail;
    throw new Error(
      (typeof detail === 'string' ? detail : null) ??
        `Fetching angle result failed with status ${res.status}`,
    );
  }

  return payload;
}

/** Trigger DTW aligned preview generation (SYNC button only). */
export async function generateDtwPreview(
  runId: string,
): Promise<GeneratePreviewResponse> {
  const res = await fetch(
    `/api/angle/learner/${encodeURIComponent(runId)}/generate-preview`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId }),
    },
  );

  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail = (payload as any)?.detail;
    throw new Error(
      (typeof detail === 'string' ? detail : null) ??
        `Preview generation failed with status ${res.status}`,
    );
  }

  return payload as GeneratePreviewResponse;
}

/** Check whether pre-computed expert angles exist for a given expert name. */
export async function getExpertAngleStatus(
  expertName: string,
): Promise<ExpertAngleStatusResponse> {
  const res = await fetch(
    `/api/angle/expert/${encodeURIComponent(expertName)}/status`,
  );

  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail = (payload as any)?.detail;
    throw new Error(
      (typeof detail === 'string' ? detail : null) ??
        `Expert status check failed with status ${res.status}`,
    );
  }

  return payload as ExpertAngleStatusResponse;
}
