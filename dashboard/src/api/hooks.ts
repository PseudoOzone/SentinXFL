import { useState, useEffect, useCallback } from 'react'
import * as api from './client'

// Generic hook for async data fetching
function useAsync<T>(
  asyncFn: () => Promise<T>,
  immediate = true
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(immediate)
  const [error, setError] = useState<Error | null>(null)

  const execute = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await asyncFn()
      setData(result)
      return result
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'))
      return null
    } finally {
      setLoading(false)
    }
  }, [asyncFn])

  useEffect(() => {
    if (immediate) {
      execute()
    }
  }, [execute, immediate])

  return { data, loading, error, execute, setData }
}

// Health check hook
export function useHealthCheck() {
  return useAsync(() => api.healthCheck())
}

// Transactions hooks
export function useTransactions(params?: {
  limit?: number
  offset?: number
  risk_level?: string
  status?: string
}) {
  const [transactions, setTransactions] = useState<api.Transaction[]>([])
  const [total, setTotal] = useState(0)
  const { loading, error } = useAsync(async () => {
    const result = await api.getTransactions(params)
    setTransactions(result.transactions)
    setTotal(result.total)
    return result
  }, true)

  const refetch = useCallback(async (newParams?: typeof params) => {
    const result = await api.getTransactions(newParams || params)
    setTransactions(result.transactions)
    setTotal(result.total)
    return result
  }, [params])

  return { transactions, total, loading, error, refetch }
}

export function useTransaction(id: string) {
  return useAsync(() => api.getTransaction(id), !!id)
}

// Fraud explanation hooks
export function useExplainTransaction() {
  const [explanation, setExplanation] = useState<api.FraudExplanation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const explain = useCallback(async (data: {
    transaction_id: string
    features: Record<string, number | string>
    fraud_probability: number
  }) => {
    setLoading(true)
    setError(null)
    try {
      const result = await api.explainTransaction(data)
      setExplanation(result)
      return result
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'))
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { explanation, loading, error, explain }
}

// RAG query hook
export function useRAGQuery() {
  const [result, setResult] = useState<api.RAGQueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const query = useCallback(async (queryText: string, topK?: number) => {
    setLoading(true)
    setError(null)
    try {
      const result = await api.queryRAG(queryText, topK)
      setResult(result)
      return result
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'))
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { result, loading, error, query }
}

// Chat hook
export function useChat() {
  const [messages, setMessages] = useState<api.ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const sendMessage = useCallback(async (
    content: string, 
    context?: Record<string, unknown>
  ) => {
    const userMessage: api.ChatMessage = { role: 'user', content }
    setMessages(prev => [...prev, userMessage])
    setLoading(true)
    setError(null)

    try {
      const allMessages = [...messages, userMessage]
      const { response } = await api.chat(allMessages, context)
      const assistantMessage: api.ChatMessage = { role: 'assistant', content: response }
      setMessages(prev => [...prev, assistantMessage])
      return response
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'))
      return null
    } finally {
      setLoading(false)
    }
  }, [messages])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return { messages, loading, error, sendMessage, clearMessages, setMessages }
}

// FL Status hooks
export function useFLStatus(pollInterval?: number) {
  const { data, loading, error, execute } = useAsync(() => api.getFLStatus(), true)

  useEffect(() => {
    if (pollInterval && pollInterval > 0) {
      const interval = setInterval(execute, pollInterval)
      return () => clearInterval(interval)
    }
  }, [pollInterval, execute])

  const startTraining = useCallback(async (config?: {
    rounds?: number
    min_clients?: number
    epsilon?: number
  }) => {
    await api.startFLTraining(config)
    await execute()
  }, [execute])

  const stopTraining = useCallback(async () => {
    await api.stopFLTraining()
    await execute()
  }, [execute])

  return { status: data, loading, error, refetch: execute, startTraining, stopTraining }
}

// Privacy hooks
export function usePrivacyMetrics() {
  return useAsync(() => api.getPrivacyMetrics())
}

export function usePrivacyAuditLog(limit = 20) {
  return useAsync(() => api.getPrivacyAuditLog(limit))
}

// Model hooks
export function usePredictFraud() {
  const [prediction, setPrediction] = useState<{ probability: number, is_fraud: boolean } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const predict = useCallback(async (features: Record<string, number>) => {
    setLoading(true)
    setError(null)
    try {
      const result = await api.predictFraud(features)
      setPrediction(result)
      return result
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'))
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { prediction, loading, error, predict }
}

export function useModelMetrics() {
  return useAsync(() => api.getModelMetrics())
}

// LLM health hook
export function useLLMHealth() {
  return useAsync(() => api.llmHealthCheck())
}
