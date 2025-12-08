import axios from 'axios'

const baseURL = import.meta.env?.VITE_API_BASE_URL ?? '/api'

export const apiClient = axios.create({
  baseURL,
  timeout: 60000,
})

export interface AnalysisPayload {
  symbols: string[]
  analysis_type?: 'basic' | 'standard' | 'detailed'
}

export async function runDetailedAnalysis(payload: AnalysisPayload) {
  const response = await apiClient.post('/analysis/quick', {
    ...payload,
    analysis_type: payload.analysis_type || 'detailed',
  })
  return response.data?.data
}

export async function fetchKnowledgeGraphArticles(symbols?: string[]) {
  const params = new URLSearchParams()
  if (symbols && symbols.length) {
    params.set('symbols', symbols.join(','))
  }
  const response = await apiClient.get(`/graph/articles${params.toString() ? `?${params.toString()}` : ''}`)
  return response.data?.data ?? []
}

export async function fetchRecentQueries(limit = 5) {
  const response = await apiClient.get('/graph/queries', { params: { limit } })
  return response.data?.data ?? []
}
