/**
 * SentinXFL - Knowledge API Client
 * 
 * API functions for knowledge module, auth, upload, and reports.
 */

const API_BASE = '/api/v1'

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('sentinxfl_token')
  const headers: HeadersInit = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function authFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: { ...getAuthHeaders(), ...options.headers },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'API error' }))
    throw new Error(err.detail || `Error ${res.status}`)
  }
  return res.json()
}

// ============================================
// Types
// ============================================

export interface PatternEntry {
  pattern_id: string
  pattern_type: string
  title: string
  description: string
  severity: string
  confidence: number
  source: string
  first_seen: string
  last_seen: string
  observation_count: number
  source_bank_count: number
  novelty_score: number
  indicators: Record<string, number>
}

export interface EmergentAlert {
  alert_id: string
  pattern_id: string
  title: string
  description: string
  severity: string
  alert_type: string
  confidence: number
  affected_banks: number
  evidence: Record<string, unknown>
  recommended_actions: string[]
  created_at: string
}

export interface BankProfile {
  bank_id: string
  display_name: string
  joined_at: string
  last_active: string
  total_transactions: number
  total_fraud_flagged: number
  avg_fraud_rate: number
  model_accuracy: number
  rounds_participated: number
  risk_score: number
}

export interface GlobalStats {
  version: number
  total_banks: number
  active_banks: number
  total_transactions_processed: number
  total_fraud_flagged: number
  global_avg_fraud_rate: number
  total_rounds: number
  pattern_library: {
    total_patterns: number
    by_type: Record<string, number>
    by_severity: Record<string, number>
  }
  last_updated: string | null
}

export interface ReportData {
  report_id: string
  report_type: string
  title: string
  generated_at: string
  generated_for: string
  summary: string
  sections: Array<{
    title: string
    content: string
    data: Record<string, unknown>
    charts: Array<{ type: string; label: string; metric?: string }>
    severity: string
  }>
}

export interface UploadInfo {
  upload_id: string
  filename: string
  bank_id: string
  file_size: number
  status: string
  uploaded_at: string
  row_count: number | null
  fraud_count: number | null
  error_message: string | null
}

// ============================================
// Knowledge API
// ============================================

export const getPatterns = (params?: { pattern_type?: string; severity?: string; limit?: number }) => {
  const sp = new URLSearchParams()
  if (params?.pattern_type) sp.set('pattern_type', params.pattern_type)
  if (params?.severity) sp.set('severity', params.severity)
  if (params?.limit) sp.set('limit', params.limit.toString())
  const q = sp.toString()
  return authFetch<{ patterns: PatternEntry[]; total: number }>(`/knowledge/patterns${q ? `?${q}` : ''}`)
}

export const getEmergentPatterns = (limit = 20) =>
  authFetch<{ patterns: PatternEntry[]; count: number }>(`/knowledge/patterns/emergent?limit=${limit}`)

export const getFactBasedPatterns = (limit = 50) =>
  authFetch<{ patterns: PatternEntry[]; count: number }>(`/knowledge/patterns/fact-based?limit=${limit}`)

export const searchPatterns = (q: string) =>
  authFetch<{ patterns: PatternEntry[]; query: string; count: number }>(`/knowledge/patterns/search?q=${encodeURIComponent(q)}`)

export const getLibraryStatistics = () =>
  authFetch<Record<string, unknown>>('/knowledge/statistics')

export const getAlerts = (params?: { severity?: string; alert_type?: string; limit?: number }) => {
  const sp = new URLSearchParams()
  if (params?.severity) sp.set('severity', params.severity)
  if (params?.alert_type) sp.set('alert_type', params.alert_type)
  if (params?.limit) sp.set('limit', params.limit.toString())
  const q = sp.toString()
  return authFetch<{ alerts: EmergentAlert[]; count: number }>(`/knowledge/alerts${q ? `?${q}` : ''}`)
}

export const getAlertSummary = () =>
  authFetch<Record<string, unknown>>('/knowledge/alerts/summary')

export const getGlobalStatistics = () =>
  authFetch<GlobalStats>('/knowledge/global/statistics')

export const getGlobalTrends = (window = 10) =>
  authFetch<Record<string, unknown>>(`/knowledge/global/trends?window=${window}`)

export const getGlobalFeatures = (topN = 20) =>
  authFetch<Array<{ feature: string; mean_importance: number; std: number; observations: number }>>(`/knowledge/global/features?top_n=${topN}`)

export const getBanks = () =>
  authFetch<{ banks: BankProfile[]; count: number }>('/knowledge/banks')

export const getBank = (bankId: string) =>
  authFetch<BankProfile>(`/knowledge/banks/${bankId}`)

export const getRiskScores = () =>
  authFetch<Record<string, number>>('/knowledge/banks/risk-scores')

// Reports
export const generateGlobalReport = () =>
  authFetch<ReportData>('/knowledge/reports/global', { method: 'POST' })

export const generateBankReport = (bankId: string) =>
  authFetch<ReportData>(`/knowledge/reports/bank/${bankId}`, { method: 'POST' })

export const generateEmergentBriefing = () =>
  authFetch<ReportData>('/knowledge/reports/emergent', { method: 'POST' })

export const generateComplianceReport = () =>
  authFetch<ReportData>('/knowledge/reports/compliance', { method: 'POST' })

export const listReports = (params?: { report_type?: string; bank_id?: string }) => {
  const sp = new URLSearchParams()
  if (params?.report_type) sp.set('report_type', params.report_type)
  if (params?.bank_id) sp.set('bank_id', params.bank_id)
  return authFetch<ReportData[]>(`/knowledge/reports?${sp.toString()}`)
}

// Upload
export const uploadFile = async (file: File, bankId: string, description = ''): Promise<UploadInfo> => {
  const token = localStorage.getItem('sentinxfl_token')
  const formData = new FormData()
  formData.append('file', file)
  formData.append('bank_id', bankId)
  formData.append('description', description)

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}

export const listUploads = (params?: { bank_id?: string; status?: string }) => {
  const sp = new URLSearchParams()
  if (params?.bank_id) sp.set('bank_id', params.bank_id)
  if (params?.status) sp.set('status', params.status)
  return authFetch<{ uploads: UploadInfo[]; count: number }>(`/upload?${sp.toString()}`)
}

export const processUpload = (uploadId: string) =>
  authFetch<UploadInfo>(`/upload/${uploadId}/process`, { method: 'POST' })

export const deleteUpload = (uploadId: string) =>
  authFetch<{ message: string }>(`/upload/${uploadId}`, { method: 'DELETE' })

// Ingestion
export const ingestRound = (data: {
  round_number: number
  bank_metrics: Record<string, Record<string, number>>
  feature_importances: Record<string, Record<string, number>>
  global_accuracy?: number
  global_loss?: number
}) =>
  authFetch<Record<string, unknown>>('/knowledge/ingest', {
    method: 'POST',
    body: JSON.stringify(data),
  })
