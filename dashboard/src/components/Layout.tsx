import { NavLink, Outlet } from 'react-router-dom'
import { 
  LayoutDashboard, 
  CreditCard, 
  Brain, 
  Shield, 
  Sparkles,
  Bell,
  Settings,
  LogOut
} from 'lucide-react'
import clsx from 'clsx'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/transactions', icon: CreditCard, label: 'Transactions' },
  { to: '/training', icon: Brain, label: 'FL Training' },
  { to: '/privacy', icon: Shield, label: 'Privacy' },
  { to: '/explainability', icon: Sparkles, label: 'Explainability' },
]

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-slate-800">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Shield className="w-8 h-8 text-primary-400" />
            <span>SentinXFL</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">Fraud Detection Platform</p>
        </div>
        
        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-1">
            {navItems.map(({ to, icon: Icon, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) => clsx(
                    'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors',
                    isActive 
                      ? 'bg-primary-600 text-white' 
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        
        {/* User section */}
        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-4 py-3">
            <div className="w-10 h-10 rounded-full bg-primary-600 flex items-center justify-center font-semibold">
              AB
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">Bank Admin</p>
              <p className="text-xs text-slate-400">Fraud Analyst</p>
            </div>
          </div>
          <button className="w-full flex items-center gap-3 px-4 py-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors mt-2">
            <LogOut className="w-4 h-4" />
            <span className="text-sm">Sign Out</span>
          </button>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
          <h2 className="text-lg font-semibold text-slate-800">
            Bank Fraud Detection System
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
        
        {/* Page content */}
        <main className="flex-1 p-6 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
