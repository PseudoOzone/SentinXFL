// Base API configuration
const API_BASE_URL = '/api/v1'

// Generic fetch wrapper with error handling
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  
  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  }
  
  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `API Error: ${response.status}`)
  }
  
  return response.json()
}

// Types
export interface Transaction {
  id: string
  amount: number
  merchant: string
  category: string
  location: string
  timestamp: string
  risk_score: number
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  status: 'pending' | 'approved' | 'flagged' | 'blocked'
}

export interface FraudExplanation {
  transaction_id: string
  risk_score: number
  risk_level: string
  summary: string
  key_factors: string[]
  feature_contributions: FeatureContribution[]
  patterns_detected: string[]
  recommendation: string
}

export interface FeatureContribution {
  feature: string
  value: number | string
  importance: number
  direction: 'positive' | 'negative'
}

export interface FLStatus {
  current_round: number
  total_rounds: number
  global_accuracy: number
  global_loss: number
  privacy_budget_used: number
  privacy_budget_total: number
  connected_banks: BankStatus[]
  status: 'idle' | 'training' | 'aggregating' | 'completed'
}

export interface BankStatus {
  id: string
  name: string
  status: 'active' | 'syncing' | 'offline'
  local_accuracy: number
  samples: number
  last_sync: string
}

export interface PrivacyMetrics {
  epsilon_current: number
  epsilon_total: number
  delta: number
  noise_multiplier: number
  clip_norm: number
  pii_fields_protected: number
  detection_rate: number
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface RAGQueryResult {
  answer: string
  sources: DocumentSource[]
}

export interface DocumentSource {
  content: string
  metadata: Record<string, unknown>
  relevance_score: number
}

// API Functions

// Health check
export const healthCheck = () => 
  fetchAPI<{ status: string }>('/health')

// Transactions
export const getTransactions = (params?: {
  limit?: number
  offset?: number
  risk_level?: string
  status?: string
}) => {
  const searchParams = new URLSearchParams()
  if (params?.limit) searchParams.set('limit', params.limit.toString())
  if (params?.offset) searchParams.set('offset', params.offset.toString())
  if (params?.risk_level) searchParams.set('risk_level', params.risk_level)
  if (params?.status) searchParams.set('status', params.status)
  
  const query = searchParams.toString()
  return fetchAPI<{ transactions: Transaction[], total: number }>(
    `/transactions${query ? `?${query}` : ''}`
  )
}

export const getTransaction = (id: string) =>
  fetchAPI<Transaction>(`/transactions/${id}`)

// Fraud Explanation
export const explainTransaction = (transactionData: {
  transaction_id: string
  features: Record<string, number | string>
  fraud_probability: number
}) =>
  fetchAPI<FraudExplanation>('/llm/explain', {
    method: 'POST',
    body: JSON.stringify(transactionData),
  })

export const batchExplainTransactions = (transactions: Array<{
  transaction_id: string
  features: Record<string, number | string>
  fraud_probability: number
}>) =>
  fetchAPI<FraudExplanation[]>('/llm/explain/batch', {
    method: 'POST',
    body: JSON.stringify({ transactions }),
  })

// RAG Query
export const queryRAG = (query: string, topK: number = 5) =>
  fetchAPI<RAGQueryResult>('/llm/rag/query', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  })

// Chat
export const chat = (messages: ChatMessage[], context?: Record<string, unknown>) =>
  fetchAPI<{ response: string }>('/llm/chat', {
    method: 'POST',
    body: JSON.stringify({ messages, context }),
  })

// LLM Health
export const llmHealthCheck = () =>
  fetchAPI<{ status: string, provider: string, model: string }>('/llm/health')

// FL Training
export const getFLStatus = () =>
  fetchAPI<FLStatus>('/fl/status')

export const startFLTraining = (config?: {
  rounds?: number
  min_clients?: number
  epsilon?: number
}) =>
  fetchAPI<{ message: string }>('/fl/start', {
    method: 'POST',
    body: JSON.stringify(config || {}),
  })

export const stopFLTraining = () =>
  fetchAPI<{ message: string }>('/fl/stop', {
    method: 'POST',
  })

// Privacy
export const getPrivacyMetrics = () =>
  fetchAPI<PrivacyMetrics>('/privacy/metrics')

export const getPrivacyAuditLog = (limit: number = 20) =>
  fetchAPI<{ logs: Array<{
    id: number
    timestamp: string
    event: string
    details: string
    severity: 'info' | 'warning' | 'success'
  }> }>(`/privacy/audit-log?limit=${limit}`)

// PII
export const processTransaction = (data: Record<string, unknown>) =>
  fetchAPI<{ processed: Record<string, unknown>, pii_detected: string[] }>(
    '/pii/process',
    {
      method: 'POST',
      body: JSON.stringify(data),
    }
  )

// Model
export const predictFraud = (features: Record<string, number>) =>
  fetchAPI<{ probability: number, is_fraud: boolean }>('/model/predict', {
    method: 'POST',
    body: JSON.stringify({ features }),
  })

export const getModelMetrics = () =>
  fetchAPI<{
    accuracy: number
    precision: number
    recall: number
    f1_score: number
    auc_roc: number
  }>('/model/metrics')
