const BASE = import.meta.env.VITE_API_URL || '/api'

async function get(path) {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export const api = {
  health: () => get('/health'),
  stats: (minutes) => get(`/stats?minutes=${minutes}`),
  predictions: (limit = 40, alertsOnly = true) => get(`/predictions?limit=${limit}&alerts_only=${alertsOnly}`),
  metrics: () => get('/model/metrics'),
  info: () => get('/model/info'),
}
