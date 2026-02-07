/**
 * Client Bank Dashboard - Overview Page
 * Shows bank-specific stats, recent alerts, and confirmed threats.
 */
import { useState, useEffect } from 'react'
import {
  Activity,
  ShieldAlert,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react'
import { Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { useAuth } from '../../contexts/AuthContext'
import * as api from '../../api/knowledge'

const severityColors: Record<string, string> = {
  low: '#22c55e',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
}

export default function ClientDashboard() {
  const { user } = useAuth()
  const [stats, setStats] = useState<api.GlobalStats | null>(null)
  const [factPatterns, setFactPatterns] = useState<api.PatternEntry[]>([])
  const [emergentPatterns, setEmergentPatterns] = useState<api.PatternEntry[]>([])
  const [alerts, setAlerts] = useState<api.EmergentAlert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [statsRes, factRes, emergentRes, alertsRes] = await Promise.all([
        api.getGlobalStatistics(),
        api.getFactBasedPatterns(10),
        api.getEmergentPatterns(5),
        api.getAlerts({ limit: 5 }),
      ])
      setStats(statsRes)
      setFactPatterns(factRes.patterns)
      setEmergentPatterns(emergentRes.patterns)
      setAlerts(alertsRes.alerts)
    } catch (e) {
      console.error('Failed to load data:', e)
    }
    setLoading(false)
  }

  const sevData = stats?.pattern_library?.by_severity
    ? Object.entries(stats.pattern_library.by_severity).map(([k, v]) => ({
        name: k,
        value: v as number,
        color: severityColors[k] || '#94a3b8',
      }))
    : []

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Welcome, {user?.display_name}</h1>
          <p className="text-slate-500 mt-1">Your fraud intelligence overview</p>
        </div>
        <button
          onClick={loadData}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Network Banks"
          value={stats?.total_banks ?? '-'}
          icon={Activity}
          color="bg-blue-500"
        />
        <StatCard
          title="Patterns Tracked"
          value={stats?.pattern_library?.total ?? '-'}
          icon={ShieldAlert}
          color="bg-purple-500"
        />
        <StatCard
          title="FL Rounds"
          value={stats?.total_rounds ?? '-'}
          icon={TrendingUp}
          color="bg-green-500"
        />
        <StatCard
          title="Active Alerts"
          value={alerts.length}
          icon={AlertTriangle}
          color="bg-red-500"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Severity Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Threat Severity</h3>
          {sevData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={sevData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                  {sevData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-slate-400">No data</div>
          )}
        </div>

        {/* Confirmed Threats */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-800">Confirmed Threat Patterns</h3>
            <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-medium">
              FACT-BASED
            </span>
          </div>
          <div className="space-y-3 max-h-[240px] overflow-y-auto">
            {factPatterns.length > 0 ? factPatterns.map((p) => (
              <div key={p.pattern_id} className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm text-slate-800 truncate">{p.title}</p>
                  <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">{p.description}</p>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  p.severity === 'critical' ? 'bg-red-100 text-red-700' :
                  p.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                  p.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-green-100 text-green-700'
                }`}>
                  {p.severity}
                </span>
              </div>
            )) : (
              <p className="text-sm text-slate-400">No confirmed patterns yet</p>
            )}
          </div>
        </div>
      </div>

      {/* Emerging Threats + Alerts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Emerging Threats */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <h3 className="text-lg font-semibold text-slate-800">Emerging Threats</h3>
          </div>
          <div className="space-y-3">
            {emergentPatterns.length > 0 ? emergentPatterns.map((p) => (
              <div key={p.pattern_id} className="p-3 border border-amber-200 bg-amber-50 rounded-lg">
                <p className="font-medium text-sm text-slate-800">{p.title}</p>
                <p className="text-xs text-slate-600 mt-1 line-clamp-2">{p.description}</p>
                <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                  <span>Confidence: {(p.confidence * 100).toFixed(0)}%</span>
                  <span>Banks: {p.source_bank_count}</span>
                </div>
              </div>
            )) : (
              <p className="text-sm text-slate-400">No emerging threats detected</p>
            )}
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <ShieldAlert className="w-5 h-5 text-red-500" />
            <h3 className="text-lg font-semibold text-slate-800">Network Alerts</h3>
          </div>
          <div className="space-y-3">
            {alerts.length > 0 ? alerts.map((a) => (
              <div key={a.alert_id} className={`p-3 rounded-lg border ${
                a.severity === 'critical' ? 'border-red-200 bg-red-50' :
                a.severity === 'high' ? 'border-orange-200 bg-orange-50' :
                'border-slate-200 bg-slate-50'
              }`}>
                <p className="font-medium text-sm text-slate-800">{a.title}</p>
                <p className="text-xs text-slate-600 mt-1 line-clamp-2">{a.description}</p>
                <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                  <span className="capitalize">{a.alert_type}</span>
                  <span>Banks: {a.affected_banks}</span>
                </div>
              </div>
            )) : (
              <p className="text-sm text-slate-400">No active alerts</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, icon: Icon, color }: { title: string; value: string | number; icon: any; color: string }) {
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
