import { 
  TrendingUp, 
  TrendingDown, 
  ShieldAlert, 
  Activity,
  Users,
  DollarSign,
  AlertTriangle,
  CheckCircle
} from 'lucide-react'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import clsx from 'clsx'

// Mock data - replace with API calls
const statsData = [
  { 
    title: 'Total Transactions', 
    value: '847,293', 
    change: '+12.5%', 
    increasing: true,
    icon: Activity,
    color: 'bg-blue-500'
  },
  { 
    title: 'Flagged Fraud', 
    value: '1,247', 
    change: '-8.3%', 
    increasing: false,
    icon: ShieldAlert,
    color: 'bg-red-500'
  },
  { 
    title: 'Amount Saved', 
    value: '$2.4M', 
    change: '+23.1%', 
    increasing: true,
    icon: DollarSign,
    color: 'bg-green-500'
  },
  { 
    title: 'Active Clients', 
    value: '12', 
    change: '+2', 
    increasing: true,
    icon: Users,
    color: 'bg-purple-500'
  },
]

const fraudTrendData = [
  { date: 'Jan', detected: 120, blocked: 115, false_positive: 8 },
  { date: 'Feb', detected: 145, blocked: 138, false_positive: 12 },
  { date: 'Mar', detected: 132, blocked: 128, false_positive: 7 },
  { date: 'Apr', detected: 168, blocked: 162, false_positive: 10 },
  { date: 'May', detected: 152, blocked: 148, false_positive: 9 },
  { date: 'Jun', detected: 189, blocked: 185, false_positive: 6 },
  { date: 'Jul', detected: 175, blocked: 172, false_positive: 5 },
]

const riskDistribution = [
  { name: 'Low', value: 68, color: '#22c55e' },
  { name: 'Medium', value: 18, color: '#eab308' },
  { name: 'High', value: 10, color: '#f97316' },
  { name: 'Critical', value: 4, color: '#ef4444' },
]

const recentAlerts = [
  { id: 'TXN-001', amount: 4582.00, risk: 'critical', time: '2 min ago', pattern: 'Card Testing' },
  { id: 'TXN-002', amount: 892.50, risk: 'high', time: '5 min ago', pattern: 'Velocity Anomaly' },
  { id: 'TXN-003', amount: 12500.00, risk: 'critical', time: '8 min ago', pattern: 'Geographic Anomaly' },
  { id: 'TXN-004', amount: 234.99, risk: 'medium', time: '12 min ago', pattern: 'New Device' },
  { id: 'TXN-005', amount: 1875.00, risk: 'high', time: '15 min ago', pattern: 'Unusual Amount' },
]

function StatCard({ stat }: { stat: typeof statsData[0] }) {
  const Icon = stat.icon
  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-500">{stat.title}</p>
          <p className="text-2xl font-bold mt-1">{stat.value}</p>
          <div className="flex items-center gap-1 mt-2">
            {stat.increasing ? (
              <TrendingUp className="w-4 h-4 text-green-500" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500" />
            )}
            <span className={clsx(
              'text-sm font-medium',
              stat.increasing ? 'text-green-600' : 'text-red-600'
            )}>
              {stat.change}
            </span>
            <span className="text-xs text-slate-400 ml-1">vs last month</span>
          </div>
        </div>
        <div className={clsx('p-3 rounded-lg', stat.color)}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  )
}

function RiskBadge({ risk }: { risk: string }) {
  return (
    <span className={clsx(
      'risk-badge',
      risk === 'low' && 'risk-low',
      risk === 'medium' && 'risk-medium',
      risk === 'high' && 'risk-high',
      risk === 'critical' && 'risk-critical',
    )}>
      {risk.charAt(0).toUpperCase() + risk.slice(1)}
    </span>
  )
}

export default function Dashboard() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Fraud Detection Dashboard</h1>
        <p className="text-slate-500 mt-1">Real-time monitoring and analytics</p>
      </div>
      
      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsData.map((stat, i) => (
          <StatCard key={i} stat={stat} />
        ))}
      </div>
      
      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fraud trend chart */}
        <div className="lg:col-span-2 card">
          <h3 className="font-semibold text-slate-800 mb-4">Fraud Detection Trend</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={fraudTrendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px'
                  }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="detected" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  name="Detected"
                  dot={{ r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="blocked" 
                  stroke="#22c55e" 
                  strokeWidth={2}
                  name="Blocked"
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Risk distribution */}
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-4">Risk Distribution</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-2 mt-4 justify-center">
            {riskDistribution.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-xs text-slate-600">{item.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Recent alerts & FL status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent alerts */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-slate-800">Recent Alerts</h3>
            <button className="text-sm text-primary-600 hover:text-primary-700 font-medium">
              View All
            </button>
          </div>
          <div className="space-y-3">
            {recentAlerts.map((alert) => (
              <div key={alert.id} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <AlertTriangle className={clsx(
                    'w-5 h-5',
                    alert.risk === 'critical' && 'text-red-500',
                    alert.risk === 'high' && 'text-orange-500',
                    alert.risk === 'medium' && 'text-yellow-500',
                  )} />
                  <div>
                    <p className="font-medium text-sm">{alert.id}</p>
                    <p className="text-xs text-slate-500">{alert.pattern}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold">${alert.amount.toLocaleString()}</p>
                  <RiskBadge risk={alert.risk} />
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* FL Training Status */}
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-4">Federated Learning Status</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-6 h-6 text-green-500" />
                <div>
                  <p className="font-medium text-green-800">Model Training Active</p>
                  <p className="text-sm text-green-600">Round 47 of 100</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-green-700">47%</p>
                <p className="text-xs text-green-600">Progress</p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-50 rounded-lg">
                <p className="text-sm text-slate-500">Connected Banks</p>
                <p className="text-xl font-bold mt-1">12 / 12</p>
              </div>
              <div className="p-4 bg-slate-50 rounded-lg">
                <p className="text-sm text-slate-500">Global Accuracy</p>
                <p className="text-xl font-bold mt-1">94.7%</p>
              </div>
              <div className="p-4 bg-slate-50 rounded-lg">
                <p className="text-sm text-slate-500">Privacy Budget (ε)</p>
                <p className="text-xl font-bold mt-1">7.2 / 10</p>
              </div>
              <div className="p-4 bg-slate-50 rounded-lg">
                <p className="text-sm text-slate-500">Last Update</p>
                <p className="text-xl font-bold mt-1">2m ago</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
