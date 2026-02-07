/**
 * Client Dashboard Layout
 * Sidebar navigation for bank client users.
 */
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  FileText,
  Shield,
  Bell,
  LogOut,
} from 'lucide-react'
import clsx from 'clsx'
import { useAuth } from '../contexts/AuthContext'

const navItems = [
  { to: '/client', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/client/upload', icon: Upload, label: 'Upload Data' },
  { to: '/client/reports', icon: FileText, label: 'Reports' },
]

export default function ClientLayout() {
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
    .toUpperCase() || 'CL'

  return (
    <div className="flex min-h-screen">
      <aside className="w-64 bg-slate-900 text-white flex flex-col">
        <div className="p-6 border-b border-slate-800">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Shield className="w-8 h-8 text-blue-400" />
            <span>SentinXFL</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">Bank Client Portal</p>
        </div>

        <nav className="flex-1 p-4">
          <ul className="space-y-1">
            {navItems.map(({ to, icon: Icon, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === '/client'}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors',
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    )
                  }
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-4 py-3">
            <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center font-semibold text-sm">
              {initials}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">{user?.display_name}</p>
              <p className="text-xs text-slate-400">{user?.bank_id || 'Client'}</p>
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
          <h2 className="text-lg font-semibold text-slate-800">Bank Fraud Intelligence</h2>
          <div className="flex items-center gap-4">
            <button className="relative p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg">
              <Bell className="w-5 h-5" />
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
