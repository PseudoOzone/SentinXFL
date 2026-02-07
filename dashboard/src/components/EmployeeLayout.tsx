/**
 * Employee Dashboard Layout
 * Sidebar navigation for SentinXFL employees with global access.
 */
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  Globe,
  Building2,
  Radio,
  Layers,
  FileText,
  Shield,
  Bell,
  LogOut,
  LayoutDashboard,
  Brain,
  Settings,
} from 'lucide-react'
import clsx from 'clsx'
import { useAuth } from '../contexts/AuthContext'

const navSections = [
  {
    title: 'Intelligence',
    items: [
      { to: '/employee', icon: Globe, label: 'Global Overview' },
      { to: '/employee/banks', icon: Building2, label: 'Bank Monitor' },
      { to: '/employee/emergent', icon: Radio, label: 'Emergent Attacks' },
      { to: '/employee/patterns', icon: Layers, label: 'Pattern Library' },
      { to: '/employee/reports', icon: FileText, label: 'Reports' },
    ],
  },
  {
    title: 'Operations',
    items: [
      { to: '/employee/ops', icon: LayoutDashboard, label: 'System Dashboard' },
      { to: '/employee/ops/training', icon: Brain, label: 'FL Training' },
      { to: '/employee/ops/privacy', icon: Shield, label: 'Privacy' },
    ],
  },
]

export default function EmployeeLayout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const initials = user?.display_name
    ?.split(' ')
    .map((n) => n[0])
    .join('')
    .slice(0, 2)
    .toUpperCase() || 'EM'

  return (
    <div className="flex min-h-screen">
      <aside className="w-64 bg-slate-900 text-white flex flex-col">
        <div className="p-6 border-b border-slate-800">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Shield className="w-8 h-8 text-emerald-400" />
            <span>SentinXFL</span>
          </h1>
          <p className="text-xs text-emerald-400 mt-1">Employee Dashboard</p>
        </div>

        <nav className="flex-1 p-4 overflow-y-auto">
          {navSections.map((section) => (
            <div key={section.title} className="mb-6">
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 mb-2">
                {section.title}
              </h3>
              <ul className="space-y-1">
                {section.items.map(({ to, icon: Icon, label }) => (
                  <li key={to}>
                    <NavLink
                      to={to}
                      end={to === '/employee' || to === '/employee/ops'}
                      className={({ isActive }) =>
                        clsx(
                          'flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm',
                          isActive
                            ? 'bg-emerald-600 text-white'
                            : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                        )
                      }
                    >
                      <Icon className="w-4 h-4" />
                      <span className="font-medium">{label}</span>
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-4 py-3">
            <div className="w-10 h-10 rounded-full bg-emerald-600 flex items-center justify-center font-semibold text-sm">
              {initials}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">{user?.display_name}</p>
              <p className="text-xs text-emerald-400">Employee</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-4 py-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors mt-2"
          >
            <LogOut className="w-4 h-4" />
            <span className="text-sm">Sign Out</span>
          </button>
        </div>
      </aside>

      <div className="flex-1 flex flex-col">
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
          <h2 className="text-lg font-semibold text-slate-800">
            SentinXFL Global Intelligence Center
          </h2>
          <div className="flex items-center gap-4">
            <button className="relative p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <button className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </header>
        <main className="flex-1 p-6 overflow-auto bg-slate-50">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
