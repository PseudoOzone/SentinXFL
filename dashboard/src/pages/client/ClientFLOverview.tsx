/**
 * Client - Federated Learning Overview
 * Shows FL round history, model convergence, bank participation, and privacy status.
 */
import { useState, useEffect } from 'react'
import {
  Brain,
  TrendingUp,
  Shield,
  Users,
  Zap,
  RefreshCw,
  CheckCircle2,
} from 'lucide-react'
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Legend,
} from 'recharts'
import { useAuth } from '../../contexts/AuthContext'
import * as api from '../../api/knowledge'

export default function ClientFLOverview() {
  const { user } = useAuth()
  const [stats, setStats] = useState<api.GlobalStats | null>(null)
  const [trends, setTrends] = useState<any>(null)
  const [banks, setBanks] = useState<api.BankProfile[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [statsRes, trendsRes, banksRes] = await Promise.all([
        api.getGlobalStatistics(),
        api.getGlobalTrends(15),
        api.getBanks(),
      ])
      setStats(statsRes)
      setTrends(trendsRes)
      setBanks(banksRes.banks)
    } catch (e) {
      console.error('Failed to load FL data:', e)
    }
    setLoading(false)
  }

  const roundHistory = trends?.rounds || []
  const accuracyData = roundHistory.map((r: any) => ({
    round: `R${r.round}`,
    accuracy: +(r.global_accuracy * 100).toFixed(1),
    loss: +r.global_loss.toFixed(3),
    banks: r.banks_participated,
  }))

  const latestRound = roundHistory.length > 0 ? roundHistory[roundHistory.length - 1] : null
  const prevRound = roundHistory.length > 1 ? roundHistory[roundHistory.length - 2] : null
  const accDelta = latestRound && prevRound
    ? +((latestRound.global_accuracy - prevRound.global_accuracy) * 100).toFixed(2)
    : 0
  const lossDelta = latestRound && prevRound
    ? +(latestRound.global_loss - prevRound.global_loss).toFixed(4)
    : 0

  // Bank participation data for bar chart
  const bankParticipation = banks
    .sort((a, b) => b.rounds_participated - a.rounds_participated)
    .slice(0, 8)
    .map((b) => ({
      name: b.display_name.length > 15 ? b.display_name.slice(0, 14) + '…' : b.display_name,
      rounds: b.rounds_participated,
      accuracy: +(b.model_accuracy * 100).toFixed(1),
    }))

  // My bank info
  const myBank = banks.find((b) => b.bank_id === user?.bank_id)

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
          <h1 className="text-2xl font-bold text-slate-800">Federated Learning</h1>
          <p className="text-slate-500 mt-1">Collaborative model training status &amp; history</p>
        </div>
        <button
          onClick={loadData}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total FL Rounds"
          value={stats?.total_rounds ?? '-'}
          icon={Brain}
          color="bg-indigo-500"
          subtitle={latestRound ? `Latest: R${latestRound.round}` : ''}
        />
        <StatCard
          title="Global Accuracy"
          value={latestRound ? `${(latestRound.global_accuracy * 100).toFixed(1)}%` : '-'}
          icon={TrendingUp}
          color="bg-green-500"
          subtitle={accDelta !== 0 ? `${accDelta > 0 ? '+' : ''}${accDelta}% from prev` : ''}
          positive={accDelta >= 0}
        />
        <StatCard
          title="Global Loss"
          value={latestRound ? latestRound.global_loss.toFixed(3) : '-'}
          icon={Zap}
          color="bg-amber-500"
          subtitle={lossDelta !== 0 ? `${lossDelta > 0 ? '+' : ''}${lossDelta} from prev` : ''}
          positive={lossDelta <= 0}
        />
        <StatCard
          title="Participating Banks"
          value={stats?.active_banks ?? stats?.total_banks ?? '-'}
          icon={Users}
          color="bg-blue-500"
          subtitle={myBank ? `You: ${myBank.rounds_participated} rounds` : ''}
        />
      </div>

      {/* My Bank FL Status */}
      {myBank && (
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3 mb-3">
            <Shield className="w-6 h-6" />
            <h3 className="text-lg font-semibold">Your FL Status &mdash; {myBank.display_name}</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-4">
            <div>
              <p className="text-blue-100 text-xs">Rounds Participated</p>
              <p className="text-3xl font-bold mt-1">{myBank.rounds_participated}</p>
            </div>
            <div>
              <p className="text-blue-100 text-xs">Local Accuracy</p>
              <p className="text-3xl font-bold mt-1">{(myBank.model_accuracy * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-blue-100 text-xs">Data Contributed</p>
              <p className="text-3xl font-bold mt-1">{(myBank.total_transactions / 1000).toFixed(0)}K</p>
            </div>
            <div>
              <p className="text-blue-100 text-xs">Risk Score</p>
              <p className="text-3xl font-bold mt-1">{myBank.risk_score.toFixed(2)}</p>
            </div>
          </div>
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Accuracy Trend */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Accuracy Over Rounds</h3>
          {accuracyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={accuracyData}>
                <defs>
                  <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="round" tick={{ fontSize: 11 }} />
                <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [`${v}%`, 'Accuracy']} />
                <Area type="monotone" dataKey="accuracy" stroke="#22c55e" fill="url(#accGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[280px] flex items-center justify-center text-slate-400">No round data</div>
          )}
        </div>

        {/* Loss Trend */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Loss Over Rounds</h3>
          {accuracyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={accuracyData}>
                <defs>
                  <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="round" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="loss" stroke="#f97316" fill="url(#lossGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[280px] flex items-center justify-center text-slate-400">No round data</div>
          )}
        </div>
      </div>

      {/* Bank Participation Bar Chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Bank Participation &amp; Local Accuracy</h3>
        {bankParticipation.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bankParticipation}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[85, 100]} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="rounds" fill="#6366f1" name="FL Rounds" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="right" dataKey="accuracy" fill="#22c55e" name="Accuracy %" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[300px] flex items-center justify-center text-slate-400">No bank data</div>
        )}
      </div>

      {/* Round History Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Recent Round History</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Round</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Accuracy</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Loss</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Banks</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Timestamp</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {[...roundHistory].reverse().map((r: any, i: number) => (
                <tr key={i} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-2 px-3 font-mono font-medium text-slate-800">R{r.round}</td>
                  <td className="py-2 px-3 text-green-600 font-medium">
                    {(r.global_accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 px-3 text-amber-600">{r.global_loss.toFixed(4)}</td>
                  <td className="py-2 px-3">
                    <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full text-xs font-medium">
                      {r.banks_participated}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-slate-500 text-xs">
                    {new Date(r.timestamp).toLocaleDateString()}
                  </td>
                  <td className="py-2 px-3">
                    <span className="flex items-center gap-1 text-green-600 text-xs">
                      <CheckCircle2 className="w-3 h-3" /> Complete
                    </span>
                  </td>
                </tr>
              ))}
              {roundHistory.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-400">
                    No FL rounds recorded yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* How FL Works Info Box */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-5 h-5 text-blue-600" />
          <h3 className="text-sm font-semibold text-blue-800">How Federated Learning Works</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-blue-700">
          <div className="flex items-start gap-2">
            <span className="bg-blue-200 text-blue-800 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">1</span>
            <p>Each bank trains a local model on its own transaction data. Raw data never leaves the bank.</p>
          </div>
          <div className="flex items-start gap-2">
            <span className="bg-blue-200 text-blue-800 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">2</span>
            <p>Only encrypted model updates are sent to SentinXFL for aggregation using secure protocols.</p>
          </div>
          <div className="flex items-start gap-2">
            <span className="bg-blue-200 text-blue-800 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">3</span>
            <p>A global model is formed combining intelligence from all banks, improving detection for everyone.</p>
          </div>
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
  subtitle,
  positive,
}: {
  title: string
  value: string | number
  icon: any
  color: string
  subtitle?: string
  positive?: boolean
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500">{title}</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{value}</p>
          {subtitle && (
            <p className={`text-xs mt-1 ${positive === false ? 'text-red-500' : positive ? 'text-green-600' : 'text-slate-500'}`}>
              {subtitle}
            </p>
          )}
        </div>
        <div className={`${color} p-3 rounded-lg`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>
    </div>
  )
}
