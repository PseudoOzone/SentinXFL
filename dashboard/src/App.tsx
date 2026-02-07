import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext'

// Layouts
import ClientLayout from './components/ClientLayout'
import EmployeeLayout from './components/EmployeeLayout'

// Auth
import Login from './pages/Login'

// Original pages (available under employee/ops)
import Dashboard from './pages/Dashboard'
import Transactions from './pages/Transactions'
import FLTraining from './pages/FLTraining'
import Privacy from './pages/Privacy'
import Explainability from './pages/Explainability'

// Client pages
import ClientDashboard from './pages/client/ClientDashboard'
import ClientUpload from './pages/client/ClientUpload'
import ClientReports from './pages/client/ClientReports'

// Employee pages
import GlobalOverview from './pages/employee/GlobalOverview'
import BankMonitor from './pages/employee/BankMonitor'
import EmergentMonitor from './pages/employee/EmergentMonitor'
import PatternManagement from './pages/employee/PatternManagement'
import GlobalReports from './pages/employee/GlobalReports'

function ProtectedRoute({ children, requiredRole }: { children: JSX.Element; requiredRole?: string }) {
  const { isAuthenticated, user, loading } = useAuth()
  
  if (loading) return <div className="flex items-center justify-center min-h-screen text-slate-500">Loading...</div>
  if (!isAuthenticated) return <Navigate to="/login" replace />
  if (requiredRole && user?.role !== requiredRole) {
    return <Navigate to={user?.role === 'employee' ? '/employee' : '/client'} replace />
  }
  return children
}

function App() {
  const { isAuthenticated, user, loading } = useAuth()

  return (
    <Routes>
      {/* Login */}
      <Route path="/login" element={<Login />} />

      {/* Root redirect */}
      <Route
        path="/"
        element={
          loading ? (
            <div className="flex items-center justify-center min-h-screen text-slate-500">Loading...</div>
          ) : !isAuthenticated ? (
            <Navigate to="/login" replace />
          ) : user?.role === 'employee' ? (
            <Navigate to="/employee" replace />
          ) : (
            <Navigate to="/client" replace />
          )
        }
      />

      {/* Client Dashboard */}
      <Route
        path="/client"
        element={
          <ProtectedRoute requiredRole="client">
            <ClientLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<ClientDashboard />} />
        <Route path="upload" element={<ClientUpload />} />
        <Route path="reports" element={<ClientReports />} />
      </Route>

      {/* Employee Dashboard */}
      <Route
        path="/employee"
        element={
          <ProtectedRoute requiredRole="employee">
            <EmployeeLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<GlobalOverview />} />
        <Route path="banks" element={<BankMonitor />} />
        <Route path="emergent" element={<EmergentMonitor />} />
        <Route path="patterns" element={<PatternManagement />} />
        <Route path="reports" element={<GlobalReports />} />
        {/* Original system pages under ops */}
        <Route path="ops" element={<Dashboard />} />
        <Route path="ops/transactions" element={<Transactions />} />
        <Route path="ops/training" element={<FLTraining />} />
        <Route path="ops/privacy" element={<Privacy />} />
        <Route path="ops/explainability" element={<Explainability />} />
      </Route>

      {/* Catch all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
