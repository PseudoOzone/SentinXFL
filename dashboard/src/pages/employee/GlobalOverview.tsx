/**
 * Employee - Global Overview Dashboard
 * Shows cross-bank intelligence, global stats, FL collaboration, and system health.
 */
import { useState, useEffect } from 'react'
import {
  Globe,
  Building2,
  ShieldAlert,
  Brain,
  AlertTriangle,
  Activity,
  Zap,
  RefreshCw,
  TrendingUp,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  Line,
} from 'recharts'
import * as api from '../../api/knowledge'

const COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#eab308', '#ef4444', '#06b6d4']

export default function EmployeeGlobalOverview() {
  const [stats, setStats] = useState<api.GlobalStats | null>(null)
  const [banks, setBanks] = useState<api.BankProfile[]>([])
  const [alerts, setAlerts] = useState<api.EmergentAlert[]>([])
  const [features, setFeatures] = useState<Array<{ feature: string; mean_importance: number }>>([])
  const [trends, setTrends] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => { loadAll() }, [])

  const loadAll = async () => {
    setLoading(true)
    try {
      const [s, b, a, f, t] = await Promise.all([
        api.getGlobalStatistics(),
        api.getBanks(),
        api.getAlerts({ limit: 10 }),
        api.getGlobalFeatures(10),
        api.getGlobalTrends(20),
      ])
      setStats(s)
      setBanks(b.banks)
      setAlerts(a.alerts)
      setFeatures(Array.isArray(f) ? f : [])
      setTrends(t)
    } catch (e) {
      console.error('Load failed:', e)
    }
    setLoading(false)
  }

  const typeData = stats?.pattern_library?.by_type
    ? Object.entries(stats.pattern_library.by_type)
        .filter(([, v]) => (v as number) > 0)
        .map(([k, v], i) => ({
          name: k,
          value: v as number,
          fill: COLORS[i % COLORS.length],
        }))
    : []

  // FL round trend data
  const roundHistory = trends?.rounds || []
  const flTrendData = roundHistory.map((r: any) => ({
    round: `R${r.round}`,
    accuracy: +(r.global_accuracy * 100).toFixed(1),
    loss: +r.global_loss.toFixed(3),
    banks: r.banks_participated,
  }))

  // Bank contribution ranking (sorted by rounds participated)
  const bankContributions = [...banks]
    .sort((a, b) => b.rounds_participated - a.rounds_participated)
    .slice(0, 8)
    .map((b) => ({
      name: b.display_name.length > 15 ? b.display_name.slice(0, 14) + '…' : b.display_name,
      rounds: b.rounds_participated,
      accuracy: +(b.model_accuracy * 100).toFixed(1),
      fraud_rate: +(b.avg_fraud_rate * 100).toFixed(2),
    }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Globe className="w-7 h-7 text-blue-600" />
            Global Intelligence Overview
          </h1>
          <p className="text-slate-500 mt-1">Cross-bank fraud intelligence center</p>
        </div>
        <button
          onClick={loadAll}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MiniStat icon={Building2} label="Total Banks" value={stats?.total_banks ?? 0} color="text-blue-600 bg-blue-100" />
        <MiniStat icon={Activity} label="Transactions" value={stats?.total_transactions_processed?.toLocaleString() ?? '0'} color="text-purple-600 bg-purple-100" />
        <MiniStat icon={ShieldAlert} label="Fraud Flagged" value={stats?.total_fraud_flagged?.toLocaleString() ?? '0'} color="text-red-600 bg-red-100" />
        <MiniStat icon={Brain} label="Patterns" value={stats?.pattern_library?.total ?? 0} color="text-green-600 bg-green-100" />
        <MiniStat icon={Zap} label="FL Rounds" value={stats?.total_rounds ?? 0} color="text-amber-600 bg-amber-100" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pattern Type Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Pattern Types</h3>
          {typeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={typeData} cx="50%" cy="50%" innerRadius={45} outerRadius={85} dataKey="value" paddingAngle={2}>
                  {typeData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number, name: string) => [value, name.charAt(0).toUpperCase() + name.slice(1)]} />
                <Legend iconType="circle" iconSize={10} wrapperStyle={{ fontSize: '12px' }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-slate-400">No pattern data</div>
          )}
        </div>

        {/* Top Features */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Top Fraud Features (Global)</h3>
          {features.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={features.slice(0, 8)} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={80} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="mean_importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-slate-400">No feature data</div>
          )}
        </div>
      </div>

      {/* FL Collaboration Intelligence */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Global Accuracy Trend */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-1 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            Global Model Performance
          </h3>
          <p className="text-xs text-slate-400 mb-4">Accuracy &amp; loss across FL rounds</p>
          {flTrendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={flTrendData}>
                <defs>
                  <linearGradient id="empAccGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="round" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="left" domain={['dataMin - 1', 100]} tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />
                <Area yAxisId="left" type="monotone" dataKey="accuracy" stroke="#22c55e" fill="url(#empAccGrad)" strokeWidth={2} name="Accuracy %" />
                <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={1.5} dot={false} name="Loss" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[260px] flex items-center justify-center text-slate-400 text-sm">No FL round data</div>
          )}
        </div>

        {/* Bank Collaboration Chart */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-1 flex items-center gap-2">
            <Building2 className="w-5 h-5 text-blue-500" />
            Bank Collaboration &amp; Contribution
          </h3>
          <p className="text-xs text-slate-400 mb-4">FL rounds participated &amp; model accuracy by bank</p>
          {bankContributions.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bankContributions}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-15} textAnchor="end" height={50} />
                <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" domain={[85, 100]} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="rounds" fill="#6366f1" name="FL Rounds" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="accuracy" fill="#22c55e" name="Accuracy %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[260px] flex items-center justify-center text-slate-400 text-sm">No bank data</div>
          )}
        </div>
      </div>

      {/* Banks Table + Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Banks Table */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200">
          <div className="p-5 border-b border-slate-100">
            <h3 className="text-lg font-semibold text-slate-800">Participating Banks</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50">
                <tr>
                  <th className="text-left px-5 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider">Bank</th>
                  <th className="text-right px-5 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider">Transactions</th>
                  <th className="text-right px-5 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider">FL Rounds</th>
                  <th className="text-right px-5 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider">Fraud Rate</th>
                  <th className="text-right px-5 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider">Accuracy</th>
                  <th className="text-right px-5 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider">Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {banks.length > 0 ? banks.map((b) => (
                  <tr key={b.bank_id} className="hover:bg-slate-50">
                    <td className="px-5 py-3">
                      <p className="font-medium text-sm text-slate-800">{b.display_name}</p>
                      <p className="text-xs text-slate-400">{b.bank_id}</p>
                    </td>
                    <td className="px-5 py-3 text-right text-sm text-slate-600">{b.total_transactions.toLocaleString()}</td>
                    <td className="px-5 py-3 text-right text-sm">
                      <span className="bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full text-xs font-medium">
                        {b.rounds_participated}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right text-sm">
                      <span className={b.avg_fraud_rate > 0.05 ? 'text-red-600 font-medium' : 'text-slate-600'}>
                        {(b.avg_fraud_rate * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right text-sm text-slate-600">{(b.model_accuracy * 100).toFixed(1)}%</td>
                    <td className="px-5 py-3 text-right">
                      <RiskBadge risk={b.risk_score} />
                    </td>
                  </tr>
                )) : (
                  <tr><td colSpan={6} className="px-5 py-8 text-center text-slate-400">No banks registered</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Active Alerts */}
        <div className="bg-white rounded-xl border border-slate-200">
          <div className="p-5 border-b border-slate-100 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <h3 className="text-lg font-semibold text-slate-800">Live Alerts</h3>
          </div>
          <div className="divide-y divide-slate-100 max-h-[400px] overflow-y-auto">
            {alerts.length > 0 ? alerts.map((a) => (
              <div key={a.alert_id} className="p-4">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`w-2 h-2 rounded-full ${
                    a.severity === 'critical' ? 'bg-red-500' :
                    a.severity === 'high' ? 'bg-orange-500' :
                    'bg-yellow-500'
                  }`} />
                  <p className="font-medium text-sm text-slate-800 truncate">{a.title}</p>
                </div>
                <p className="text-xs text-slate-500 line-clamp-2">{a.description}</p>
                <div className="flex items-center gap-2 mt-2 text-xs text-slate-400">
                  <span className="capitalize">{a.alert_type}</span>
                  <span>·</span>
                  <span>{a.affected_banks} banks</span>
                </div>
              </div>
            )) : (
              <div className="p-8 text-center text-slate-400 text-sm">No active alerts</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function MiniStat({ icon: Icon, label, value, color }: { icon: any; label: string; value: string | number; color: string }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${color}`}>
          <Icon className="w-4 h-4" />
        </div>
        <div>
          <p className="text-xs text-slate-500">{label}</p>
          <p className="text-lg font-bold text-slate-800">{value}</p>
        </div>
      </div>
    </div>
  )
}

function RiskBadge({ risk }: { risk: number }) {
  const color = risk > 0.7 ? 'bg-red-100 text-red-700' :
                risk > 0.4 ? 'bg-amber-100 text-amber-700' :
                'bg-green-100 text-green-700'
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${color}`}>
      {(risk * 100).toFixed(0)}%
    </span>
  )
}
