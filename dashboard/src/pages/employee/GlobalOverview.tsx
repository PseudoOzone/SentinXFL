/**
 * Employee - Global Overview Dashboard
 * Shows cross-bank intelligence, global stats, and system health.
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
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import * as api from '../../api/knowledge'

const COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#eab308', '#ef4444', '#06b6d4']

export default function EmployeeGlobalOverview() {
  const [stats, setStats] = useState<api.GlobalStats | null>(null)
  const [banks, setBanks] = useState<api.BankProfile[]>([])
  const [alerts, setAlerts] = useState<api.EmergentAlert[]>([])
  const [features, setFeatures] = useState<Array<{ feature: string; mean_importance: number }>>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => { loadAll() }, [])

  const loadAll = async () => {
    setLoading(true)
    try {
      const [s, b, a, f] = await Promise.all([
        api.getGlobalStatistics(),
        api.getBanks(),
        api.getAlerts({ limit: 10 }),
        api.getGlobalFeatures(10),
      ])
      setStats(s)
      setBanks(b.banks)
      setAlerts(a.alerts)
      setFeatures(Array.isArray(f) ? f : [])
    } catch (e) {
      console.error('Load failed:', e)
    }
    setLoading(false)
  }

  const typeData = stats?.pattern_library?.by_type
    ? Object.entries(stats.pattern_library.by_type).map(([k, v], i) => ({
        name: k,
        value: v as number,
        fill: COLORS[i % COLORS.length],
      }))
    : []

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
                <Pie data={typeData} cx="50%" cy="50%" outerRadius={90} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                  {typeData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
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
                  <tr><td colSpan={5} className="px-5 py-8 text-center text-slate-400">No banks registered</td></tr>
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
