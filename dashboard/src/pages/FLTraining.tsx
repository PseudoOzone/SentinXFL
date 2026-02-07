import { 
  Activity, 
  Server, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp,
  Shield,
  Zap
} from 'lucide-react'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts'
import clsx from 'clsx'

// Mock FL training data
const trainingProgress = {
  currentRound: 47,
  totalRounds: 100,
  globalAccuracy: 94.7,
  globalLoss: 0.128,
  privacyBudgetUsed: 7.2,
  privacyBudgetTotal: 10,
  lastUpdate: '2 minutes ago',
  status: 'training'
}

const connectedBanks = [
  { id: 'BANK_001', name: 'First National Bank', status: 'active', localAccuracy: 93.2, samples: 125000, lastSync: '30s ago' },
  { id: 'BANK_002', name: 'Metro Credit Union', status: 'active', localAccuracy: 95.1, samples: 89000, lastSync: '45s ago' },
  { id: 'BANK_003', name: 'Pacific Trust', status: 'active', localAccuracy: 94.8, samples: 156000, lastSync: '20s ago' },
  { id: 'BANK_004', name: 'Atlantic Finance', status: 'active', localAccuracy: 92.4, samples: 78000, lastSync: '1m ago' },
  { id: 'BANK_005', name: 'Central Bank Corp', status: 'syncing', localAccuracy: 94.5, samples: 203000, lastSync: 'syncing...' },
  { id: 'BANK_006', name: 'Pioneer Savings', status: 'active', localAccuracy: 96.1, samples: 67000, lastSync: '15s ago' },
]

const trainingHistory = [
  { round: 40, accuracy: 92.1, loss: 0.185 },
  { round: 41, accuracy: 92.8, loss: 0.172 },
  { round: 42, accuracy: 93.2, loss: 0.163 },
  { round: 43, accuracy: 93.6, loss: 0.155 },
  { round: 44, accuracy: 94.0, loss: 0.148 },
  { round: 45, accuracy: 94.3, loss: 0.139 },
  { round: 46, accuracy: 94.5, loss: 0.132 },
  { round: 47, accuracy: 94.7, loss: 0.128 },
]

const aggregationMethods = [
  { name: 'FedAvg', description: 'Federated Averaging', active: true },
  { name: 'Multi-Krum', description: 'Byzantine-resilient', active: true },
  { name: 'Trimmed Mean', description: 'Outlier robust', active: false },
  { name: 'Coordinate Median', description: 'Dimension-wise median', active: false },
]

function BankStatusCard({ bank }: { bank: typeof connectedBanks[0] }) {
  return (
    <div className={clsx(
      'p-4 rounded-lg border transition-all',
      bank.status === 'active' && 'bg-green-50 border-green-200',
      bank.status === 'syncing' && 'bg-yellow-50 border-yellow-200',
      bank.status === 'offline' && 'bg-red-50 border-red-200',
    )}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Server className={clsx(
            'w-5 h-5',
            bank.status === 'active' && 'text-green-600',
            bank.status === 'syncing' && 'text-yellow-600',
            bank.status === 'offline' && 'text-red-600',
          )} />
          <div>
            <p className="font-medium text-slate-900">{bank.name}</p>
            <p className="text-xs text-slate-500 font-mono">{bank.id}</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {bank.status === 'active' && <CheckCircle className="w-4 h-4 text-green-500" />}
          {bank.status === 'syncing' && <Clock className="w-4 h-4 text-yellow-500 animate-spin" />}
          {bank.status === 'offline' && <XCircle className="w-4 h-4 text-red-500" />}
          <span className={clsx(
            'text-xs font-medium capitalize',
            bank.status === 'active' && 'text-green-600',
            bank.status === 'syncing' && 'text-yellow-600',
            bank.status === 'offline' && 'text-red-600',
          )}>
            {bank.status}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4 mt-4">
        <div>
          <p className="text-xs text-slate-500">Local Accuracy</p>
          <p className="text-lg font-bold text-slate-900">{bank.localAccuracy}%</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Data Samples</p>
          <p className="text-lg font-bold text-slate-900">{(bank.samples / 1000).toFixed(0)}K</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Last Sync</p>
          <p className="text-sm font-medium text-slate-700">{bank.lastSync}</p>
        </div>
      </div>
    </div>
  )
}

export default function FLTraining() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Federated Learning</h1>
          <p className="text-slate-500 mt-1">Distributed model training status</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-2 text-green-600">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span className="text-sm font-medium">Training Active</span>
          </span>
          <button className="btn-secondary">Pause Training</button>
        </div>
      </div>
      
      {/* Training Progress */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="card bg-gradient-to-br from-primary-600 to-primary-700 text-white">
          <div className="flex items-center gap-3 mb-3">
            <Activity className="w-6 h-6" />
            <span className="font-medium">Training Round</span>
          </div>
          <p className="text-4xl font-bold">{trainingProgress.currentRound}</p>
          <p className="text-primary-200 mt-1">of {trainingProgress.totalRounds} rounds</p>
          <div className="mt-4 bg-primary-500/30 rounded-full h-2">
            <div 
              className="bg-white rounded-full h-2 transition-all"
              style={{ width: `${(trainingProgress.currentRound / trainingProgress.totalRounds) * 100}%` }}
            />
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center gap-3 mb-3 text-green-600">
            <TrendingUp className="w-6 h-6" />
            <span className="font-medium text-slate-600">Global Accuracy</span>
          </div>
          <p className="text-4xl font-bold text-slate-900">{trainingProgress.globalAccuracy}%</p>
          <p className="text-green-600 text-sm mt-1">+2.1% from last round</p>
        </div>
        
        <div className="card">
          <div className="flex items-center gap-3 mb-3 text-orange-600">
            <Zap className="w-6 h-6" />
            <span className="font-medium text-slate-600">Global Loss</span>
          </div>
          <p className="text-4xl font-bold text-slate-900">{trainingProgress.globalLoss}</p>
          <p className="text-green-600 text-sm mt-1">-0.004 from last round</p>
        </div>
        
        <div className="card">
          <div className="flex items-center gap-3 mb-3 text-purple-600">
            <Shield className="w-6 h-6" />
            <span className="font-medium text-slate-600">Privacy Budget (ε)</span>
          </div>
          <p className="text-4xl font-bold text-slate-900">{trainingProgress.privacyBudgetUsed}</p>
          <p className="text-slate-500 text-sm mt-1">of {trainingProgress.privacyBudgetTotal} total</p>
          <div className="mt-4 bg-slate-200 rounded-full h-2">
            <div 
              className="bg-purple-500 rounded-full h-2"
              style={{ width: `${(trainingProgress.privacyBudgetUsed / trainingProgress.privacyBudgetTotal) * 100}%` }}
            />
          </div>
        </div>
      </div>
      
      {/* Training History Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-4">Training Accuracy Progress</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trainingHistory}>
                <defs>
                  <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="round" stroke="#64748b" fontSize={12} />
                <YAxis domain={[90, 100]} stroke="#64748b" fontSize={12} />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#22c55e" 
                  fillOpacity={1} 
                  fill="url(#colorAccuracy)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-4">Training Loss Progress</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="round" stroke="#64748b" fontSize={12} />
                <YAxis domain={[0, 0.25]} stroke="#64748b" fontSize={12} />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#f97316" 
                  strokeWidth={2}
                  dot={{ r: 4, fill: '#f97316' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* Connected Banks */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-semibold text-slate-800">Connected Banks</h3>
          <div className="flex items-center gap-4 text-sm">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span className="text-slate-600">{connectedBanks.filter(b => b.status === 'active').length} Active</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
              <span className="text-slate-600">{connectedBanks.filter(b => b.status === 'syncing').length} Syncing</span>
            </span>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {connectedBanks.map((bank) => (
            <BankStatusCard key={bank.id} bank={bank} />
          ))}
        </div>
      </div>
      
      {/* Aggregation Methods */}
      <div className="card">
        <h3 className="font-semibold text-slate-800 mb-4">Aggregation Strategy</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {aggregationMethods.map((method) => (
            <div 
              key={method.name}
              className={clsx(
                'p-4 rounded-lg border-2 transition-all',
                method.active 
                  ? 'border-primary-500 bg-primary-50' 
                  : 'border-slate-200 bg-slate-50'
              )}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-slate-900">{method.name}</span>
                {method.active && <CheckCircle className="w-4 h-4 text-primary-600" />}
              </div>
              <p className="text-xs text-slate-500">{method.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
