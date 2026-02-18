/**
 * Client - My Bank Page
 * Bank-specific profile with indicators, model performance, and metrics.
 */
import { useState, useEffect } from 'react'
import {
  Building2,
  TrendingUp,
  ShieldAlert,
  Activity,
  BarChart3,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from 'recharts'
import { useAuth } from '../../contexts/AuthContext'
import * as api from '../../api/knowledge'

const severityColors: Record<string, string> = {
  low: '#22c55e',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
}

export default function ClientMyBank() {
  const { user } = useAuth()
  const [bank, setBank] = useState<api.BankProfile | null>(null)
  const [patterns, setPatterns] = useState<api.PatternEntry[]>([])
  const [features, setFeatures] = useState<Array<{ feature: string; mean_importance: number }>>([])
  const [trends, setTrends] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const bankId = user?.bank_id
      const [bankRes, patternsRes, featuresRes, trendsRes] = await Promise.all([
        bankId ? api.getBank(bankId).catch(() => null) : Promise.resolve(null),
        api.getFactBasedPatterns(20),
        api.getGlobalFeatures(15),
        api.getGlobalTrends(15),
      ])
      setBank(bankRes)
      setPatterns(patternsRes.patterns)
      setFeatures(featuresRes)
      setTrends(trendsRes)
    } catch (e: any) {
      setError(e.message || 'Failed to load bank data')
    }
    setLoading(false)
  }

  const featureBarData = features.slice(0, 10).map((f) => ({
    name: f.feature.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    importance: +(f.mean_importance * 100).toFixed(1),
  }))

  const severityBreakdown = patterns.reduce(
    (acc, p) => {
      acc[p.severity] = (acc[p.severity] || 0) + 1
      return acc
    },
    {} as Record<string, number>
  )
  const sevPieData = Object.entries(severityBreakdown)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({
      name: k.charAt(0).toUpperCase() + k.slice(1),
      value: v,
      color: severityColors[k] || '#94a3b8',
    }))

  const roundHistory = trends?.rounds || []
  const accuracyData = roundHistory.map((r: any) => ({
    round: `R${r.round}`,
    accuracy: +(r.global_accuracy * 100).toFixed(1),
    loss: +r.global_loss.toFixed(3),
  }))

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-6 h-6 text-blue-500 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">My Bank Profile</h1>
          <p className="text-slate-500 mt-1">
            {bank?.display_name || user?.bank_id || 'Your bank'} &mdash; Detailed intelligence view
          </p>
        </div>
        <button
          onClick={loadData}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Bank Stats Cards */}
      {bank && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Transactions"
            value={bank.total_transactions.toLocaleString()}
            icon={Activity}
            color="bg-blue-500"
          />
          <StatCard
            title="Fraud Flagged"
            value={bank.total_fraud_flagged.toLocaleString()}
            icon={ShieldAlert}
            color="bg-red-500"
          />
          <StatCard
            title="Model Accuracy"
            value={`${(bank.model_accuracy * 100).toFixed(1)}%`}
            icon={TrendingUp}
            color="bg-green-500"
          />
          <StatCard
            title="Risk Score"
            value={bank.risk_score.toFixed(2)}
            icon={AlertTriangle}
            color={bank.risk_score > 0.5 ? 'bg-red-500' : bank.risk_score > 0.3 ? 'bg-amber-500' : 'bg-green-500'}
          />
        </div>
      )}

      {/* Bank Detail Info */}
      {bank && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Building2 className="w-6 h-6 text-blue-500" />
            <h3 className="text-lg font-semibold text-slate-800">Bank Details</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-slate-500">Bank ID</p>
              <p className="font-medium text-slate-800 font-mono">{bank.bank_id}</p>
            </div>
            <div>
              <p className="text-slate-500">Fraud Rate</p>
              <p className="font-medium text-slate-800">{(bank.avg_fraud_rate * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-slate-500">FL Rounds Participated</p>
              <p className="font-medium text-slate-800">{bank.rounds_participated}</p>
            </div>
            <div>
              <p className="text-slate-500">Last Active</p>
              <p className="font-medium text-slate-800">{new Date(bank.last_active).toLocaleDateString()}</p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Importance */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-purple-500" />
            <h3 className="text-lg font-semibold text-slate-800">Top Feature Indicators</h3>
          </div>
          {featureBarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureBarData} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={130} />
                <Tooltip formatter={(v: number) => [`${v}%`, 'Importance']} />
                <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-400">No feature data</div>
          )}
        </div>

        {/* Threat Severity Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Threat Severity Distribution</h3>
          {sevPieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={sevPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  dataKey="value"
                  paddingAngle={2}
                >
                  {sevPieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend iconType="circle" iconSize={10} wrapperStyle={{ fontSize: '12px' }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-400">No data</div>
          )}
        </div>
      </div>

      {/* Model Accuracy Trend */}
      {accuracyData.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Model Accuracy Trend (Global)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="round" tick={{ fontSize: 11 }} />
              <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => [`${v}%`, 'Accuracy']} />
              <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Known Patterns Relevant to Bank */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-800">Relevant Threat Patterns</h3>
          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium">
            {patterns.length} patterns tracked
          </span>
        </div>
        <div className="space-y-3 max-h-[400px] overflow-y-auto">
          {patterns.map((p) => (
            <div key={p.pattern_id} className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
              <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm text-slate-800">{p.name}</p>
                <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">{p.description}</p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {p.tags?.slice(0, 4).map((tag, i) => (
                    <span key={i} className="text-[10px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${
                  p.severity === 'critical' ? 'bg-red-100 text-red-700' :
                  p.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                  p.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-green-100 text-green-700'
                }`}
              >
                {p.severity}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({
  title,
  value,
  icon: Icon,
  color,
}: {
  title: string
  value: string | number
  icon: any
  color: string
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500">{title}</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{value}</p>
        </div>
        <div className={`${color} p-3 rounded-lg`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>
    </div>
  )
}
