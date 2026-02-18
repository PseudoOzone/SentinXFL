import { useState } from 'react'

interface GeneratorConfig {
  n_samples: number
  fraud_ratio: number
  n_features: number
  include_pii: boolean
  dataset_name: string
  save: boolean
  format: string
  seed: number
}

interface GeneratedDataset {
  dataset_name: string
  total_rows: number
  total_columns: number
  numeric_features: number
  categorical_features: number
  fraud_count: number
  legitimate_count: number
  fraud_ratio: number
  memory_mb: number
  columns: string[]
  sample_rows: Record<string, unknown>[]
  file_path: string | null
}

const presets: Record<string, Partial<GeneratorConfig>> = {
  quick_test: { n_samples: 1000, fraud_ratio: 0.1, n_features: 15, dataset_name: 'quick_test' },
  balanced: { n_samples: 10000, fraud_ratio: 0.15, n_features: 20, dataset_name: 'balanced_fraud' },
  realistic: { n_samples: 50000, fraud_ratio: 0.035, n_features: 25, dataset_name: 'realistic_fraud' },
  large_scale: { n_samples: 200000, fraud_ratio: 0.02, n_features: 30, dataset_name: 'large_scale_fraud' },
  privacy_test: { n_samples: 5000, fraud_ratio: 0.05, n_features: 20, include_pii: true, dataset_name: 'privacy_test' },
}

export default function DatasetGenerator() {
  const [config, setConfig] = useState<GeneratorConfig>({
    n_samples: 10000,
    fraud_ratio: 0.05,
    n_features: 20,
    include_pii: false,
    dataset_name: 'synthetic_fraud',
    save: true,
    format: 'csv',
    seed: 42,
  })

  const [result, setResult] = useState<GeneratedDataset | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activePreset, setActivePreset] = useState<string | null>(null)

  const applyPreset = (name: string) => {
    const preset = presets[name]
    if (preset) {
      setConfig(prev => ({ ...prev, ...preset }))
      setActivePreset(name)
    }
  }

  const generate = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/v1/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || `Error ${response.status}`)
      }

      const data: GeneratedDataset = await response.json()
      setResult(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate dataset')
    } finally {
      setLoading(false)
    }
  }

  const formatValue = (v: unknown): string => {
    if (v === null || v === undefined) return '-'
    if (typeof v === 'number') return Number.isInteger(v) ? v.toString() : v.toFixed(4)
    return String(v)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dataset Generator</h1>
        <p className="mt-1 text-sm text-gray-600">
          Generate synthetic fraud detection datasets for training and testing.
        </p>
      </div>

      {/* Presets */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-sm font-semibold text-gray-700 mb-3">Quick Presets</h2>
        <div className="flex flex-wrap gap-2">
          {Object.entries(presets).map(([name, preset]) => (
            <button
              key={name}
              onClick={() => applyPreset(name)}
              className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                activePreset === name
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {name.replace(/_/g, ' ')}
              <span className="ml-1 opacity-60">
                ({(preset.n_samples ?? 0).toLocaleString()} rows)
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Dataset Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset Name
            </label>
            <input
              type="text"
              value={config.dataset_name}
              onChange={e => setConfig(p => ({ ...p, dataset_name: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>

          {/* Number of Samples */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Samples
            </label>
            <input
              type="number"
              min={100}
              max={1000000}
              step={1000}
              value={config.n_samples}
              onChange={e => setConfig(p => ({ ...p, n_samples: parseInt(e.target.value) || 1000 }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-indigo-500 focus:border-indigo-500"
            />
            <p className="mt-1 text-xs text-gray-500">100 – 1,000,000</p>
          </div>

          {/* Fraud Ratio */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Fraud Ratio: {(config.fraud_ratio * 100).toFixed(1)}%
            </label>
            <input
              type="range"
              min={0.1}
              max={50}
              step={0.1}
              value={config.fraud_ratio * 100}
              onChange={e => setConfig(p => ({ ...p, fraud_ratio: parseFloat(e.target.value) / 100 }))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0.1%</span>
              <span>50%</span>
            </div>
          </div>

          {/* Number of Features */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Features: {config.n_features}
            </label>
            <input
              type="range"
              min={10}
              max={50}
              step={1}
              value={config.n_features}
              onChange={e => setConfig(p => ({ ...p, n_features: parseInt(e.target.value) }))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>10</span>
              <span>50</span>
            </div>
          </div>

          {/* Seed */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Random Seed
            </label>
            <input
              type="number"
              min={0}
              value={config.seed}
              onChange={e => setConfig(p => ({ ...p, seed: parseInt(e.target.value) || 0 }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>

          {/* Format */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              File Format
            </label>
            <select
              value={config.format}
              onChange={e => setConfig(p => ({ ...p, format: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="csv">CSV</option>
              <option value="parquet">Parquet</option>
            </select>
          </div>
        </div>

        {/* Toggles */}
        <div className="flex items-center gap-6 mt-6 pt-4 border-t border-gray-200">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.include_pii}
              onChange={e => setConfig(p => ({ ...p, include_pii: e.target.checked }))}
              className="h-4 w-4 text-indigo-600 rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">Include PII columns</span>
            <span className="text-xs text-yellow-600">(for privacy testing)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.save}
              onChange={e => setConfig(p => ({ ...p, save: e.target.checked }))}
              className="h-4 w-4 text-indigo-600 rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">Save to disk</span>
          </label>
        </div>

        {/* Generate Button */}
        <div className="mt-6">
          <button
            onClick={generate}
            disabled={loading}
            className={`px-6 py-2.5 rounded-lg text-white font-medium text-sm transition-all ${
              loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 shadow hover:shadow-md'
            }`}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Generating...
              </span>
            ) : (
              'Generate Dataset'
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Generated: {result.dataset_name}
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Total Rows" value={result.total_rows.toLocaleString()} color="blue" />
              <StatCard label="Features" value={result.total_columns.toString()} color="purple" />
              <StatCard
                label="Fraud Count"
                value={result.fraud_count.toLocaleString()}
                sub={`${(result.fraud_ratio * 100).toFixed(2)}%`}
                color="red"
              />
              <StatCard
                label="Legitimate"
                value={result.legitimate_count.toLocaleString()}
                sub={`${((1 - result.fraud_ratio) * 100).toFixed(2)}%`}
                color="green"
              />
            </div>

            {/* Fraud ratio bar */}
            <div className="mt-4">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>Class Distribution</span>
                <span>{result.memory_mb} MB</span>
              </div>
              <div className="w-full bg-green-200 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-red-500 h-3 rounded-l-full transition-all"
                  style={{ width: `${Math.max(result.fraud_ratio * 100, 1)}%` }}
                />
              </div>
              <div className="flex justify-between text-xs mt-1">
                <span className="text-red-600">Fraud ({(result.fraud_ratio * 100).toFixed(1)}%)</span>
                <span className="text-green-600">Legitimate ({((1 - result.fraud_ratio) * 100).toFixed(1)}%)</span>
              </div>
            </div>

            {result.file_path && (
              <div className="mt-4 p-3 bg-gray-50 rounded-md">
                <p className="text-xs text-gray-500">Saved to:</p>
                <p className="text-sm font-mono text-gray-700 break-all">{result.file_path}</p>
              </div>
            )}
          </div>

          {/* Columns */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              Columns ({result.columns.length})
            </h3>
            <div className="flex flex-wrap gap-1.5">
              {result.columns.map(col => (
                <span
                  key={col}
                  className={`px-2 py-0.5 text-xs rounded-full font-mono ${
                    col === 'is_fraud'
                      ? 'bg-red-100 text-red-700'
                      : col === 'transaction_id' || col === 'timestamp'
                      ? 'bg-blue-100 text-blue-700'
                      : col.startsWith('feature_')
                      ? 'bg-gray-100 text-gray-600'
                      : 'bg-indigo-50 text-indigo-700'
                  }`}
                >
                  {col}
                </span>
              ))}
            </div>
          </div>

          {/* Sample Data */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Sample Rows</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    {result.columns.slice(0, 12).map(col => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                    {result.columns.length > 12 && (
                      <th className="px-3 py-2 text-left font-medium text-gray-400">
                        +{result.columns.length - 12} more
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {result.sample_rows.map((row, i) => (
                    <tr key={i} className={row.is_fraud === 1 ? 'bg-red-50' : ''}>
                      {result.columns.slice(0, 12).map(col => (
                        <td key={col} className="px-3 py-1.5 whitespace-nowrap text-gray-700">
                          {col === 'is_fraud' ? (
                            <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                              row[col] === 1 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                            }`}>
                              {row[col] === 1 ? 'FRAUD' : 'LEGIT'}
                            </span>
                          ) : (
                            formatValue(row[col])
                          )}
                        </td>
                      ))}
                      {result.columns.length > 12 && <td className="px-3 py-1.5 text-gray-400">...</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string
  value: string
  sub?: string
  color: 'blue' | 'purple' | 'red' | 'green'
}) {
  const colors = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    green: 'bg-green-50 text-green-700 border-green-200',
  }
  return (
    <div className={`rounded-lg border p-3 ${colors[color]}`}>
      <p className="text-xs font-medium opacity-70">{label}</p>
      <p className="text-xl font-bold mt-0.5">{value}</p>
      {sub && <p className="text-xs opacity-60 mt-0.5">{sub}</p>}
    </div>
  )
}
