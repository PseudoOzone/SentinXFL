/**
 * SettingsModal - User preferences and account settings
 * Used in both ClientLayout and EmployeeLayout headers.
 */
import { useState, useEffect, useRef } from 'react'
import { Settings, X, User, Bell, Eye, Shield, Moon, Sun, Monitor } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

type Theme = 'light' | 'dark' | 'system'

interface UserSettings {
  theme: Theme
  notifications_enabled: boolean
  email_alerts: boolean
  sound_enabled: boolean
  auto_refresh: boolean
  refresh_interval: number // seconds
  compact_view: boolean
}

const defaultSettings: UserSettings = {
  theme: 'light',
  notifications_enabled: true,
  email_alerts: false,
  sound_enabled: true,
  auto_refresh: true,
  refresh_interval: 30,
  compact_view: false,
}

function loadSettings(): UserSettings {
  try {
    const saved = localStorage.getItem('sentinxfl_settings')
    if (saved) return { ...defaultSettings, ...JSON.parse(saved) }
  } catch {}
  return defaultSettings
}

function saveSettings(settings: UserSettings) {
  localStorage.setItem('sentinxfl_settings', JSON.stringify(settings))
}

export default function SettingsModal() {
  const [open, setOpen] = useState(false)
  const [settings, setSettings] = useState<UserSettings>(loadSettings)
  const [saved, setSaved] = useState(false)
  const { user } = useAuth()
  const modalRef = useRef<HTMLDivElement>(null)

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    if (open) document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [open])

  const update = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    setSettings((prev) => {
      const next = { ...prev, [key]: value }
      saveSettings(next)
      return next
    })
    setSaved(true)
    setTimeout(() => setSaved(false), 1500)
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
      >
        <Settings className="w-5 h-5" />
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div
            ref={modalRef}
            className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <Settings className="w-5 h-5 text-slate-600" />
                <h2 className="text-lg font-semibold text-slate-800">Settings</h2>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-600 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="px-6 py-5 space-y-6 max-h-[70vh] overflow-y-auto">
              {/* Account Info */}
              <section>
                <div className="flex items-center gap-2 mb-3">
                  <User className="w-4 h-4 text-slate-500" />
                  <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Account</h3>
                </div>
                <div className="bg-slate-50 rounded-lg p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Name</span>
                    <span className="text-slate-800 font-medium">{user?.display_name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Email</span>
                    <span className="text-slate-800 font-medium">{user?.email}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Role</span>
                    <span className="capitalize text-slate-800 font-medium">{user?.role}</span>
                  </div>
                  {user?.bank_id && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Bank ID</span>
                      <span className="text-slate-800 font-mono text-xs">{user.bank_id}</span>
                    </div>
                  )}
                </div>
              </section>

              {/* Appearance */}
              <section>
                <div className="flex items-center gap-2 mb-3">
                  <Eye className="w-4 h-4 text-slate-500" />
                  <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Appearance</h3>
                </div>
                <div className="flex gap-2">
                  {(['light', 'dark', 'system'] as Theme[]).map((t) => (
                    <button
                      key={t}
                      onClick={() => update('theme', t)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                        settings.theme === t
                          ? 'bg-blue-50 border-blue-300 text-blue-700'
                          : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      {t === 'light' && <Sun className="w-4 h-4" />}
                      {t === 'dark' && <Moon className="w-4 h-4" />}
                      {t === 'system' && <Monitor className="w-4 h-4" />}
                      <span className="capitalize">{t}</span>
                    </button>
                  ))}
                </div>
                <ToggleRow
                  label="Compact view"
                  description="Reduce spacing and font sizes"
                  checked={settings.compact_view}
                  onChange={(v) => update('compact_view', v)}
                />
              </section>

              {/* Notifications */}
              <section>
                <div className="flex items-center gap-2 mb-3">
                  <Bell className="w-4 h-4 text-slate-500" />
                  <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Notifications</h3>
                </div>
                <div className="space-y-1">
                  <ToggleRow
                    label="Enable notifications"
                    description="Show alert notifications in the header"
                    checked={settings.notifications_enabled}
                    onChange={(v) => update('notifications_enabled', v)}
                  />
                  <ToggleRow
                    label="Email alerts"
                    description="Receive critical alerts via email"
                    checked={settings.email_alerts}
                    onChange={(v) => update('email_alerts', v)}
                  />
                  <ToggleRow
                    label="Sound"
                    description="Play sound for new alerts"
                    checked={settings.sound_enabled}
                    onChange={(v) => update('sound_enabled', v)}
                  />
                </div>
              </section>

              {/* Data & Refresh */}
              <section>
                <div className="flex items-center gap-2 mb-3">
                  <Shield className="w-4 h-4 text-slate-500" />
                  <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Data & Refresh</h3>
                </div>
                <ToggleRow
                  label="Auto-refresh dashboards"
                  description="Automatically refresh data at intervals"
                  checked={settings.auto_refresh}
                  onChange={(v) => update('auto_refresh', v)}
                />
                {settings.auto_refresh && (
                  <div className="mt-2 ml-1">
                    <label className="text-xs text-slate-500 block mb-1">Refresh interval</label>
                    <select
                      value={settings.refresh_interval}
                      onChange={(e) => update('refresh_interval', Number(e.target.value))}
                      className="text-sm border border-slate-200 rounded-lg px-3 py-1.5 text-slate-700 focus:ring-2 focus:ring-blue-300 focus:border-blue-400"
                    >
                      <option value={15}>15 seconds</option>
                      <option value={30}>30 seconds</option>
                      <option value={60}>1 minute</option>
                      <option value={300}>5 minutes</option>
                    </select>
                  </div>
                )}
              </section>
            </div>

            {/* Footer */}
            <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
              <span className={`text-xs transition-opacity ${saved ? 'text-green-600 opacity-100' : 'opacity-0'}`}>
                Settings saved
              </span>
              <button
                onClick={() => setOpen(false)}
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div>
        <p className="text-sm text-slate-700 font-medium">{label}</p>
        <p className="text-xs text-slate-400">{description}</p>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-5 rounded-full transition-colors ${
          checked ? 'bg-blue-500' : 'bg-slate-300'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
            checked ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  )
}
