/**
 * Employee - Real-Time Emergent Attacks Monitor
 * Live view of emergent attacks, zero-day patterns, and threat intelligence.
 */
import { useState, useEffect } from 'react'
import {
  Zap,
  AlertTriangle,
  ShieldAlert,
  Radio,
  RefreshCw,
  ChevronRight,
  Target,
} from 'lucide-react'
import * as api from '../../api/knowledge'

export default function EmergentMonitor() {
  const [alerts, setAlerts] = useState<api.EmergentAlert[]>([])
  const [emergentPatterns, setEmergentPatterns] = useState<api.PatternEntry[]>([])
  const [selectedAlert, setSelectedAlert] = useState<api.EmergentAlert | null>(null)
  const [alertSummary, setAlertSummary] = useState<Record<string, unknown>>({})
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(false)

  useEffect(() => { loadData() }, [])

  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(loadData, 10000) // 10s refresh
    return () => clearInterval(interval)
  }, [autoRefresh])

  const loadData = async () => {
    setLoading(true)
    try {
      const [alertsRes, patternsRes, summaryRes] = await Promise.all([
        api.getAlerts({ limit: 50 }),
        api.getEmergentPatterns(20),
        api.getAlertSummary(),
      ])
      setAlerts(alertsRes.alerts)
      setEmergentPatterns(patternsRes.patterns)
      setAlertSummary(summaryRes)
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const totalAlerts = (alertSummary as any)?.total_alerts ?? 0
  const bySeverity = (alertSummary as any)?.by_severity ?? {}

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Radio className="w-7 h-7 text-red-500" />
            Emergent Attack Monitor
          </h1>
          <p className="text-slate-500 mt-1">Real-time threat detection and zero-day tracking</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-slate-300"
            />
            Auto-refresh
          </label>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Alert Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatCard label="Total Alerts" value={totalAlerts} color="bg-slate-100 text-slate-700" />
        <StatCard label="Critical" value={bySeverity.critical ?? 0} color="bg-red-100 text-red-700" />
        <StatCard label="High" value={bySeverity.high ?? 0} color="bg-orange-100 text-orange-700" />
        <StatCard label="Medium" value={bySeverity.medium ?? 0} color="bg-yellow-100 text-yellow-700" />
        <StatCard label="Emergent Patterns" value={emergentPatterns.length} color="bg-purple-100 text-purple-700" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert Feed */}
        <div className="lg:col-span-2 space-y-3">
          <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-500" />
            Alert Feed
          </h2>
          {alerts.length > 0 ? alerts.map((a) => (
            <div
              key={a.alert_id}
              onClick={() => setSelectedAlert(a)}
              className={`bg-white rounded-xl border p-4 cursor-pointer transition-all hover:shadow-md ${
                selectedAlert?.alert_id === a.alert_id ? 'border-red-500 ring-1 ring-red-200' :
                a.severity === 'critical' ? 'border-red-200' :
                a.severity === 'high' ? 'border-orange-200' :
                'border-slate-200'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 ${
                  a.severity === 'critical' ? 'bg-red-500 animate-pulse' :
                  a.severity === 'high' ? 'bg-orange-500' :
                  'bg-yellow-500'
                }`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-semibold text-sm text-slate-800">{a.title}</p>
                    <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                      a.alert_type === 'zero_day' ? 'bg-red-100 text-red-700' :
                      a.alert_type === 'correlation' ? 'bg-purple-100 text-purple-700' :
                      a.alert_type === 'spike' ? 'bg-amber-100 text-amber-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {a.alert_type}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 line-clamp-2">{a.description}</p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-slate-400">
                    <span>Confidence: {(a.confidence * 100).toFixed(0)}%</span>
                    <span>{a.affected_banks} bank(s) affected</span>
                    <span>{new Date(a.created_at).toLocaleTimeString()}</span>
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-300 flex-shrink-0 mt-1" />
              </div>
            </div>
          )) : (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
              <ShieldAlert className="w-10 h-10 text-green-400 mx-auto mb-3" />
              <p className="text-slate-500 font-medium">No Active Alerts</p>
              <p className="text-sm text-slate-400 mt-1">The system is operating normally</p>
            </div>
          )}
        </div>

        {/* Alert Detail Panel */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 sticky top-6 self-start">
          {selectedAlert ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  selectedAlert.severity === 'critical' ? 'bg-red-500 animate-pulse' : 'bg-orange-500'
                }`} />
                <span className="text-xs font-medium text-slate-500 uppercase">{selectedAlert.severity} ALERT</span>
              </div>

              <h3 className="text-lg font-bold text-slate-800">{selectedAlert.title}</h3>
              <p className="text-sm text-slate-600">{selectedAlert.description}</p>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-xs text-slate-500">Type</p>
                  <p className="font-medium text-slate-800 capitalize">{selectedAlert.alert_type}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Confidence</p>
                  <p className="font-medium text-slate-800">{(selectedAlert.confidence * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Affected Banks</p>
                  <p className="font-medium text-slate-800">{selectedAlert.affected_banks}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Detected</p>
                  <p className="font-medium text-slate-800">{new Date(selectedAlert.created_at).toLocaleString()}</p>
                </div>
              </div>

              {selectedAlert.recommended_actions.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-slate-500 mb-2">Recommended Actions</p>
                  <ul className="space-y-1.5">
                    {selectedAlert.recommended_actions.map((action, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                        <Target className="w-3.5 h-3.5 text-blue-500 mt-0.5 flex-shrink-0" />
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedAlert.evidence && Object.keys(selectedAlert.evidence).length > 0 && (
                <details>
                  <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-700">
                    Evidence ({Object.keys(selectedAlert.evidence).length} fields)
                  </summary>
                  <pre className="mt-2 text-xs bg-slate-50 p-3 rounded border overflow-x-auto">
                    {JSON.stringify(selectedAlert.evidence, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-400">
              <AlertTriangle className="w-10 h-10 mx-auto mb-3" />
              <p>Select an alert to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className={`${color} rounded-xl p-4`}>
      <p className="text-xs font-medium opacity-70">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  )
}
