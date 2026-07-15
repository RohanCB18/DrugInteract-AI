const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };

  const response = await fetch(url, config);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  predict: (drugA, drugB, drugAName, drugBName) =>
    request('/predict', {
      method: 'POST',
      body: JSON.stringify({
        drug_a: drugA,
        drug_b: drugB,
        drug_a_name: drugAName || null,
        drug_b_name: drugBName || null,
      }),
    }),

  getComparison: () => request('/compare'),

  getHistory: (limit = 50) => request(`/history?limit=${limit}`),

  getDrugs: () => request('/drugs'),

  getLabels: () => request('/labels'),

  healthCheck: () => request('/health'),
};
