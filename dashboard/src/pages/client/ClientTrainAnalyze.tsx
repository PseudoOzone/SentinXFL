/**
 * Client - Train & Analyze Page
 * 
 * Full pipeline: select dataset → configure FL training → run training →
 * view accuracy/loss charts → see detected attacks → contribute to global intelligence.
 */
import { useState, useEffect } from 'react'
import {
  Brain,
  TrendingUp,
  Shield,
  AlertTriangle,
  Play,
  Loader2,
  CheckCircle2,
  BarChart3,
  Target,
  Globe,
  FileText,
  ChevronDown,
  ChevronUp,
  Zap,
  Database,
  RefreshCw,
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Line,
  BarChart,
  Bar,
  Legend,
} from 'recharts'
import { useAuth } from '../../contexts/AuthContext'
import * as api from '../../api/knowledge'

const STRATEGIES = [
  { value: 'fedavg', label: 'FedAvg', desc: 'Standard weighted averaging' },
  { value: 'krum', label: 'Multi-Krum', desc: 'Byzantine-resilient selection' },
  { value: 'trimmed_mean', label: 'Trimmed Mean', desc: 'Coordinate-wise trimmed mean' },
  { value: 'median', label: 'Median', desc: 'Coordinate-wise median' },
]

const severityColor: Record<string, string> = {
  critical: 'bg-red-100 text-red-700 border-red-200',
  high: 'bg-orange-100 text-orange-700 border-orange-200',
  medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  low: 'bg-green-100 text-green-700 border-green-200',
}

const severityBg: Record<string, string> = {
  critical: 'border-l-red-500 bg-red-50',
  high: 'border-l-orange-500 bg-orange-50',
  medium: 'border-l-yellow-500 bg-yellow-50',
  low: 'border-l-green-500 bg-green-50',
}

export default function ClientTrainAnalyze() {
  const { user } = useAuth()

  // Available datasets
  const [datasets, setDatasets] = useState<api.AvailableDataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(true)

  // Training config
  const [numClients, setNumClients] = useState(3)
  const [numRounds, setNumRounds] = useState(10)
  const [strategy, setStrategy] = useState('fedavg')
  const [dpEnabled, setDpEnabled] = useState(true)
  const [dpEpsilon, setDpEpsilon] = useState(1.0)
  const [maxRows, setMaxRows] = useState(50000)

  // Training state
  const [training, setTraining] = useState(false)
  const [result, setResult] = useState<api.TrainAnalyzeResponse | null>(null)
  const [error, setError] = useState('')

  // Expanded sections
  const [expandedPattern, setExpandedPattern] = useState<number | null>(null)

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    setLoadingDatasets(true)
    try {
      const res = await api.getAvailableDatasets()
      setDatasets(res.datasets)
      if (res.datasets.length > 0) {
        setSelectedDataset(res.datasets[0].filename)
      }
    } catch (e) {
      console.error('Failed to load datasets:', e)
    }
    setLoadingDatasets(false)
  }

  const startTraining = async () => {
    if (!selectedDataset || !user?.bank_id) return

    setTraining(true)
    setError('')
    setResult(null)

    try {
      const response = await api.trainAndAnalyze({
        dataset_path: selectedDataset,
        bank_id: user.bank_id,
        bank_name: user.display_name || user.bank_id,
        num_clients: numClients,
        num_rounds: numRounds,
        aggregation_strategy: strategy,
        dp_enabled: dpEnabled,
        dp_epsilon: dpEpsilon,
        max_rows: maxRows,
      })
      setResult(response)
    } catch (e: any) {
      setError(e.message || 'Training failed')
    }
    setTraining(false)
  }

  // Chart data
  const accuracyData = result?.round_results.map((r) => ({
    round: `R${r.round}`,
    accuracy: +(r.accuracy * 100).toFixed(1),
    f1: +(r.f1 * 100).toFixed(1),
  })) || []

  const lossData = result?.round_results.map((r) => ({
    round: `R${r.round}`,
    loss: +r.loss.toFixed(4),
    epsilon: r.privacy_spent != null ? +r.privacy_spent.toFixed(3) : undefined,
  })) || []

  // Feature importance data from detected patterns
  const allFeatures: Record<string, number> = {}
  result?.detected_patterns.forEach((p) => {
    Object.entries(p.top_features).forEach(([k, v]) => {
      allFeatures[k] = Math.max(allFeatures[k] || 0, v)
    })
  })
  const featureData = Object.entries(allFeatures)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, value]) => ({
      name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      importance: +(value * 100).toFixed(1),
    }))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <Brain className="w-7 h-7 text-blue-600" />
          Train &amp; Analyze
        </h1>
        <p className="text-slate-500 mt-1">
          Run federated learning on your dataset, detect attacks, and contribute to global intelligence
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">
          {error}
        </div>
      )}

      {/* ─── Configuration Panel ──────────────────────── */}
      {!result && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-500" />
            Training Configuration
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {/* Dataset Selection */}
            <div className="lg:col-span-2">
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
              {loadingDatasets ? (
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <Loader2 className="w-4 h-4 animate-spin" /> Loading datasets...
                </div>
              ) : (
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {datasets.map((d) => (
                    <option key={d.filename} value={d.filename}>
                      {d.name} ({d.size_mb.toFixed(1)} MB)
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Max Rows */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Max Rows</label>
              <input
                type="number"
                value={maxRows}
                onChange={(e) => setMaxRows(+e.target.value)}
                min={100}
                max={500000}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm"
              />
            </div>

            {/* Number of Clients */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                FL Clients (virtual banks)
              </label>
              <input
                type="range"
                min={2}
                max={10}
                value={numClients}
                onChange={(e) => setNumClients(+e.target.value)}
                className="w-full"
              />
              <span className="text-xs text-slate-500">{numClients} clients</span>
            </div>

            {/* Number of Rounds */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Training Rounds
              </label>
              <input
                type="range"
                min={3}
                max={30}
                value={numRounds}
                onChange={(e) => setNumRounds(+e.target.value)}
                className="w-full"
              />
              <span className="text-xs text-slate-500">{numRounds} rounds</span>
            </div>

            {/* Aggregation Strategy */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Aggregation Strategy
              </label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.value} value={s.value}>{s.label} — {s.desc}</option>
                ))}
              </select>
            </div>

            {/* Differential Privacy */}
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
                <input
                  type="checkbox"
                  checked={dpEnabled}
                  onChange={(e) => setDpEnabled(e.target.checked)}
                  className="rounded"
                />
                Differential Privacy
              </label>
              {dpEnabled && (
                <div>
                  <label className="text-xs text-slate-500">Epsilon Budget: {dpEpsilon}</label>
                  <input
                    type="range"
                    min={0.1}
                    max={10}
                    step={0.1}
                    value={dpEpsilon}
                    onChange={(e) => setDpEpsilon(+e.target.value)}
                    className="w-full"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Start Button */}
          <div className="mt-6 flex items-center gap-4">
            <button
              onClick={startTraining}
              disabled={training || !selectedDataset}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium transition-colors"
            >
              {training ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Training in progress...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Start Training &amp; Analysis
                </>
              )}
            </button>
            {training && (
              <p className="text-sm text-slate-500 animate-pulse">
                Running FL simulation on {selectedDataset}...
              </p>
            )}
          </div>
        </div>
      )}

      {/* ─── Results ──────────────────────────────────── */}
      {result && (
        <>
          {/* Summary Banner */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="w-6 h-6" />
                  <h2 className="text-xl font-bold">Training Complete</h2>
                </div>
                <p className="text-blue-100 text-sm">
                  Dataset: <span className="font-semibold text-white">{result.dataset_name}</span>
                  {' · '}{result.dataset_rows.toLocaleString()} rows
                  {' · '}{result.dataset_fraud_count.toLocaleString()} fraud ({(result.dataset_fraud_ratio * 100).toFixed(1)}%)
                </p>
              </div>
              <button
                onClick={() => setResult(null)}
                className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg text-sm font-medium transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                New Training
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-5">
              <MetricCard label="Final Accuracy" value={`${(result.final_accuracy * 100).toFixed(1)}%`} />
              <MetricCard label="Final F1 Score" value={`${(result.final_f1 * 100).toFixed(1)}%`} />
              <MetricCard label="Final Loss" value={result.final_loss.toFixed(4)} />
              <MetricCard label="FL Rounds" value={result.num_rounds} />
              <MetricCard label="Privacy ε" value={result.final_epsilon?.toFixed(2) ?? 'N/A'} />
            </div>
          </div>

          {/* Accuracy + Loss Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Accuracy Over Rounds */}
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-1 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-500" />
                Accuracy Over Rounds
              </h3>
              <p className="text-xs text-slate-400 mb-4">Dataset: {result.dataset_name}</p>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={accuracyData}>
                  <defs>
                    <linearGradient id="accGradTrain" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="f1GradTrain" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="round" tick={{ fontSize: 11 }} />
                  <YAxis domain={['dataMin - 2', 100]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number, name: string) => [`${v}%`, name === 'accuracy' ? 'Accuracy' : 'F1 Score']} />
                  <Legend />
                  <Area type="monotone" dataKey="accuracy" stroke="#22c55e" fill="url(#accGradTrain)" strokeWidth={2} name="Accuracy" />
                  <Area type="monotone" dataKey="f1" stroke="#3b82f6" fill="url(#f1GradTrain)" strokeWidth={2} name="F1 Score" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Loss Over Rounds */}
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-1 flex items-center gap-2">
                <Zap className="w-5 h-5 text-orange-500" />
                Loss Over Rounds
              </h3>
              <p className="text-xs text-slate-400 mb-4">Dataset: {result.dataset_name}</p>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={lossData}>
                  <defs>
                    <linearGradient id="lossGradTrain" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="round" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="loss" stroke="#f97316" fill="url(#lossGradTrain)" strokeWidth={2} name="Loss" />
                  {dpEnabled && (
                    <Line type="monotone" dataKey="epsilon" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="ε Spent" />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detected Attacks / Patterns */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                Attacks Detected in <span className="text-blue-600">{result.dataset_name}</span>
              </h3>
              <span className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded-full font-medium">
                {result.detected_patterns.length} pattern{result.detected_patterns.length !== 1 ? 's' : ''}
              </span>
            </div>

            <div className="space-y-3">
              {result.detected_patterns.map((pattern, i) => (
                <div
                  key={i}
                  className={`border-l-4 rounded-lg overflow-hidden ${severityBg[pattern.severity] || 'border-l-slate-400 bg-slate-50'}`}
                >
                  <button
                    onClick={() => setExpandedPattern(expandedPattern === i ? null : i)}
                    className="w-full p-4 flex items-center justify-between text-left hover:bg-black/5 transition-colors"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <Target className="w-5 h-5 text-slate-600 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-slate-800">{pattern.name}</p>
                        <p className="text-xs text-slate-500 mt-0.5 truncate">{pattern.attack_vector}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${severityColor[pattern.severity] || 'bg-slate-100 text-slate-600'}`}>
                        {pattern.severity.toUpperCase()}
                      </span>
                      <span className="text-xs text-slate-500">
                        {(pattern.confidence * 100).toFixed(0)}% confidence
                      </span>
                      {expandedPattern === i ? (
                        <ChevronUp className="w-4 h-4 text-slate-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-slate-400" />
                      )}
                    </div>
                  </button>

                  {expandedPattern === i && (
                    <div className="px-4 pb-4 space-y-3 border-t border-black/10 pt-3">
                      <p className="text-sm text-slate-600">{pattern.description}</p>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="bg-white rounded-lg p-3 border">
                          <p className="text-xs text-slate-500">Source Dataset</p>
                          <p className="font-semibold text-sm text-slate-800">{result.dataset_name}</p>
                        </div>
                        <div className="bg-white rounded-lg p-3 border">
                          <p className="text-xs text-slate-500">Observations</p>
                          <p className="font-semibold text-sm text-slate-800">{pattern.observation_count.toLocaleString()}</p>
                        </div>
                        <div className="bg-white rounded-lg p-3 border">
                          <p className="text-xs text-slate-500">Confidence</p>
                          <p className="font-semibold text-sm text-slate-800">{(pattern.confidence * 100).toFixed(0)}%</p>
                        </div>
                        <div className="bg-white rounded-lg p-3 border">
                          <p className="text-xs text-slate-500">Attack Vector</p>
                          <p className="font-semibold text-sm text-slate-800">{pattern.attack_vector.replace(/_/g, ' ')}</p>
                        </div>
                      </div>

                      {/* Feature Indicators */}
                      <div>
                        <p className="text-xs font-medium text-slate-500 mb-2">Top Feature Indicators</p>
                        <div className="space-y-1.5">
                          {Object.entries(pattern.top_features)
                            .sort((a, b) => b[1] - a[1])
                            .map(([feat, imp]) => (
                              <div key={feat} className="flex items-center gap-2">
                                <span className="text-xs text-slate-600 w-40 truncate">{feat.replace(/_/g, ' ')}</span>
                                <div className="flex-1 bg-slate-200 rounded-full h-2">
                                  <div
                                    className="bg-blue-500 h-2 rounded-full"
                                    style={{ width: `${Math.min(100, imp * 500)}%` }}
                                  />
                                </div>
                                <span className="text-xs text-slate-500 w-12 text-right">{(imp * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Feature Importance Chart */}
          {featureData.length > 0 && (
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-500" />
                Top Fraud Indicators — {result.dataset_name}
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureData} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={150} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Importance']} />
                  <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Global Intelligence Contribution */}
          <div className={`rounded-xl border p-6 ${
            result.intelligence_ingested
              ? 'bg-emerald-50 border-emerald-200'
              : 'bg-slate-50 border-slate-200'
          }`}>
            <div className="flex items-center gap-3 mb-4">
              <Globe className={`w-6 h-6 ${result.intelligence_ingested ? 'text-emerald-600' : 'text-slate-400'}`} />
              <h3 className="text-lg font-semibold text-slate-800">Global Intelligence Contribution</h3>
            </div>

            {result.intelligence_ingested ? (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white rounded-lg p-4 border border-emerald-200">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    <p className="text-xs text-slate-500">Status</p>
                  </div>
                  <p className="text-lg font-bold text-emerald-700">Ingested</p>
                  <p className="text-xs text-slate-500 mt-1">Results shared with global model</p>
                </div>
                <div className="bg-white rounded-lg p-4 border border-emerald-200">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-purple-500" />
                    <p className="text-xs text-slate-500">Global Model Version</p>
                  </div>
                  <p className="text-lg font-bold text-slate-800">v{result.global_model_version}</p>
                  <p className="text-xs text-slate-500 mt-1">Updated with your data</p>
                </div>
                <div className="bg-white rounded-lg p-4 border border-emerald-200">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-4 h-4 text-blue-500" />
                    <p className="text-xs text-slate-500">New Patterns Mined</p>
                  </div>
                  <p className="text-lg font-bold text-slate-800">{result.new_patterns_mined}</p>
                  <p className="text-xs text-slate-500 mt-1">Added to pattern library</p>
                </div>
                <div className="bg-white rounded-lg p-4 border border-emerald-200">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                    <p className="text-xs text-slate-500">Alerts Generated</p>
                  </div>
                  <p className="text-lg font-bold text-slate-800">{result.new_alerts_generated}</p>
                  <p className="text-xs text-slate-500 mt-1">Network-wide alerts</p>
                </div>
              </div>
            ) : (
              <div className="text-center py-4">
                <p className="text-sm text-slate-500">
                  Intelligence ingestion was not available. Results are local only.
                </p>
              </div>
            )}

            {result.intelligence_ingested && (
              <div className="mt-4 bg-white rounded-lg p-4 border border-emerald-200">
                <div className="flex items-start gap-3">
                  <FileText className="w-5 h-5 text-emerald-600 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-slate-800">
                      Your training results from <span className="font-semibold text-blue-600">{result.dataset_name}</span> have been shared with the SentinXFL global intelligence network.
                    </p>
                    <ul className="mt-2 space-y-1 text-xs text-slate-600">
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                        Feature importances aggregated into global model
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                        Bank profile updated (accuracy: {(result.final_accuracy * 100).toFixed(1)}%, rounds: +1)
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                        Pattern mining completed — {result.detected_patterns.length} patterns analyzed
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                        Differential Privacy applied — raw data never leaves your bank
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Round Details Table */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">
              Per-Round Training Details — {result.dataset_name}
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">Round</th>
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">Accuracy</th>
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">F1 Score</th>
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">Loss</th>
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">Clients</th>
                    {dpEnabled && (
                      <th className="text-left py-2 px-3 text-slate-500 font-medium">ε Spent</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {result.round_results.map((r) => (
                    <tr key={r.round} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-2 px-3 font-mono font-medium text-slate-800">R{r.round}</td>
                      <td className="py-2 px-3 text-green-600 font-medium">
                        {(r.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-blue-600">
                        {(r.f1 * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-amber-600">{r.loss.toFixed(4)}</td>
                      <td className="py-2 px-3">
                        <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full text-xs font-medium">
                          {r.clients_active}
                        </span>
                      </td>
                      {dpEnabled && (
                        <td className="py-2 px-3 text-purple-600 font-mono text-xs">
                          {r.privacy_spent != null ? r.privacy_spent.toFixed(3) : '—'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white/10 rounded-lg px-4 py-3">
      <p className="text-blue-100 text-xs">{label}</p>
      <p className="text-2xl font-bold mt-0.5">{value}</p>
    </div>
  )
}
