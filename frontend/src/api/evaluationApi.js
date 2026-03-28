const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

async function parseJsonResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const message =
      (payload && (payload.detail || payload.message)) ||
      `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return payload || {};
}

export async function startEvaluation(formData) {
  const requestOptions =
    formData instanceof FormData
      ? {
          method: 'POST',
          body: formData,
        }
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData || {}),
        };

  const response = await fetch(`${API_BASE}/evaluations/start`, requestOptions);
  const payload = await parseJsonResponse(response);

  if (payload.status === 'out_of_context') {
    return payload;
  }

  return payload.evaluation_id || payload.id || payload;
}

export async function getEvaluationResult(id) {
  const primaryResponse = await fetch(`${API_BASE}/evaluations/${id}`);
  if (primaryResponse.ok) {
    return parseJsonResponse(primaryResponse);
  }
  if (primaryResponse.status === 404) {
    const fallbackResponse = await fetch(`${API_BASE}/evaluations/${id}/result`);
    return parseJsonResponse(fallbackResponse);
  }
  return parseJsonResponse(primaryResponse);
}

