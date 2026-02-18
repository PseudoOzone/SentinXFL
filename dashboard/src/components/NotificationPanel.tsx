/**
 * NotificationPanel - Dropdown panel for alert notifications
 * Used in both ClientLayout and EmployeeLayout headers.
 */
import { useState, useEffect, useRef } from 'react'
import { Bell, AlertTriangle, ShieldAlert, Radio, X, CheckCircle2 } from 'lucide-react'
import * as api from '../api/knowledge'

const severityIcon: Record<string, typeof AlertTriangle> = {
  critical: ShieldAlert,
  high: AlertTriangle,
  medium: Radio,
  low: CheckCircle2,
}

const severityColor: Record<string, string> = {
  critical: 'text-red-500',
  high: 'text-orange-500',
  medium: 'text-amber-500',
  low: 'text-green-500',
}

const severityBg: Record<string, string> = {
  critical: 'bg-red-50 border-red-200',
  high: 'bg-orange-50 border-orange-200',
  medium: 'bg-amber-50 border-amber-200',
  low: 'bg-green-50 border-green-200',
}

export default function NotificationPanel() {
  const [open, setOpen] = useState(false)
  const [alerts, setAlerts] = useState<api.EmergentAlert[]>([])
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  // Close on click outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  // Fetch alerts when panel opens
  useEffect(() => {
    if (open && alerts.length === 0) {
      loadAlerts()
    }
  }, [open])

  const loadAlerts = async () => {
    setLoading(true)
    try {
      const res = await api.getAlerts({ limit: 15 })
      setAlerts(res.alerts)
    } catch (e) {
      console.error('Failed to load notifications:', e)
    }
    setLoading(false)
  }

  const dismiss = (alertId: string) => {
    setDismissed((prev) => new Set([...prev, alertId]))
  }

  const dismissAll = () => {
    setDismissed(new Set(alerts.map((a) => a.alert_id)))
  }

  const visibleAlerts = alerts.filter((a) => !dismissed.has(a.alert_id))
  const unreadCount = visibleAlerts.length

  return (
    <div ref={panelRef} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="relative p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
      >
        <Bell className="w-5 h-5" />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center">
            <span className="text-[10px] text-white font-bold">
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-12 w-96 bg-white rounded-xl border border-slate-200 shadow-xl z-50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-slate-50">
            <h3 className="font-semibold text-slate-800 text-sm">Notifications</h3>
            <div className="flex items-center gap-2">
              {visibleAlerts.length > 0 && (
                <button
                  onClick={dismissAll}
                  className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                >
                  Dismiss all
                </button>
              )}
              <button
                onClick={() => setOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Notification List */}
          <div className="max-h-96 overflow-y-auto">
            {loading ? (
              <div className="p-6 text-center text-slate-400 text-sm">Loading...</div>
            ) : visibleAlerts.length === 0 ? (
              <div className="p-8 text-center">
                <CheckCircle2 className="w-8 h-8 text-green-400 mx-auto mb-2" />
                <p className="text-slate-500 text-sm">All caught up!</p>
                <p className="text-slate-400 text-xs mt-1">No new notifications</p>
              </div>
            ) : (
              visibleAlerts.map((alert) => {
                const Icon = severityIcon[alert.severity] || AlertTriangle
                return (
                  <div
                    key={alert.alert_id}
                    className={`px-4 py-3 border-b border-slate-100 hover:bg-slate-50 transition-colors`}
                  >
                    <div className="flex items-start gap-3">
                      <Icon className={`w-4 h-4 mt-0.5 flex-shrink-0 ${severityColor[alert.severity] || 'text-slate-400'}`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-800 line-clamp-1">{alert.title}</p>
                        <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">{alert.description}</p>
                        <div className="flex items-center gap-2 mt-1.5">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium capitalize ${severityBg[alert.severity] || 'bg-slate-100'} border`}>
                            {alert.severity}
                          </span>
                          <span className="text-[10px] text-slate-400">{alert.affected_banks} banks affected</span>
                        </div>
                      </div>
                      <button
                        onClick={() => dismiss(alert.alert_id)}
                        className="p-1 text-slate-300 hover:text-slate-500 flex-shrink-0"
                        title="Dismiss"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                )
              })
            )}
          </div>

          {/* Footer */}
          {visibleAlerts.length > 0 && (
            <div className="px-4 py-2 border-t border-slate-100 bg-slate-50">
              <button
                onClick={loadAlerts}
                className="text-xs text-blue-600 hover:text-blue-800 font-medium w-full text-center"
              >
                Refresh notifications
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
