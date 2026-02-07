import {
  Shield,
  Lock,
  Eye,
  EyeOff,
  AlertTriangle,
  CheckCircle,
  Info,
  TrendingDown,
  History
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts'
import clsx from 'clsx'

// Privacy metrics
const privacyMetrics = {
  epsilonCurrent: 7.2,
  epsilonTotal: 10,
  deltaCurrent: 1e-5,
  noiseMultiplier: 1.1,
  clipNorm: 1.0,
  sensitiveFieldsProtected: 12,
  piiDetectionRate: 99.8,
  dataAnonymizationRate: 100
}

const epsilonHistory = [
  { round: 1, epsilon: 0.5, cumulative: 0.5 },
  { round: 10, epsilon: 0.3, cumulative: 2.1 },
  { round: 20, epsilon: 0.25, cumulative: 3.8 },
  { round: 30, epsilon: 0.2, cumulative: 5.2 },
  { round: 40, epsilon: 0.15, cumulative: 6.4 },
  { round: 47, epsilon: 0.12, cumulative: 7.2 },
]

const piiFields = [
  { field: 'Credit Card Number', technique: 'Tokenization', status: 'protected' },
  { field: 'SSN', technique: 'Hashing + Salt', status: 'protected' },
  { field: 'Account Number', technique: 'Format-Preserving Encryption', status: 'protected' },
  { field: 'Phone Number', technique: 'Masking', status: 'protected' },
  { field: 'Email Address', technique: 'Pseudonymization', status: 'protected' },
  { field: 'Full Name', technique: 'K-Anonymity', status: 'protected' },
  { field: 'IP Address', technique: 'IP Masking', status: 'protected' },
  { field: 'Date of Birth', technique: 'Generalization', status: 'protected' },
]

const auditLogs = [
  { id: 1, timestamp: '2025-01-15 10:32:45', event: 'Privacy budget consumption', details: 'ε=0.12 consumed in round 47', severity: 'info' },
  { id: 2, timestamp: '2025-01-15 10:30:12', event: 'Model gradients clipped', details: 'Clip norm: 1.0 applied to 3 banks', severity: 'info' },
  { id: 3, timestamp: '2025-01-15 10:28:33', event: 'PII detected and protected', details: '15 credit card numbers tokenized', severity: 'warning' },
  { id: 4, timestamp: '2025-01-15 10:25:00', event: 'Noise injection completed', details: 'Gaussian noise (σ=1.1) added', severity: 'info' },
  { id: 5, timestamp: '2025-01-15 10:20:15', event: 'Secure aggregation verified', details: 'All gradients encrypted', severity: 'success' },
]

const noiseDistribution = [
  { bin: '-3σ', count: 2 },
  { bin: '-2σ', count: 15 },
  { bin: '-1σ', count: 68 },
  { bin: '0', count: 95 },
  { bin: '1σ', count: 70 },
  { bin: '2σ', count: 14 },
  { bin: '3σ', count: 3 },
]

function MetricCard({ 
  icon: Icon, 
  label, 
  value, 
  subtext, 
  color 
}: { 
  icon: typeof Shield, 
  label: string, 
  value: string | number, 
  subtext: string,
  color: string 
}) {
  return (
    <div className="card">
      <div className={clsx('flex items-center gap-3 mb-3', color)}>
        <Icon className="w-6 h-6" />
        <span className="font-medium text-slate-600">{label}</span>
      </div>
      <p className="text-3xl font-bold text-slate-900">{value}</p>
      <p className="text-slate-500 text-sm mt-1">{subtext}</p>
    </div>
  )
}

export default function Privacy() {
  const epsilonUsagePercent = (privacyMetrics.epsilonCurrent / privacyMetrics.epsilonTotal) * 100
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Privacy Dashboard</h1>
          <p className="text-slate-500 mt-1">Differential privacy & data protection metrics</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-2 text-green-600">
            <Lock className="w-4 h-4" />
            <span className="text-sm font-medium">All Data Protected</span>
          </span>
        </div>
      </div>
      
      {/* Privacy Budget Alert */}
      {epsilonUsagePercent > 70 && (
        <div className={clsx(
          'p-4 rounded-lg flex items-center gap-3',
          epsilonUsagePercent > 90 ? 'bg-red-50 border border-red-200' : 'bg-yellow-50 border border-yellow-200'
        )}>
          <AlertTriangle className={clsx(
            'w-5 h-5',
            epsilonUsagePercent > 90 ? 'text-red-600' : 'text-yellow-600'
          )} />
          <div>
            <p className={clsx(
              'font-medium',
              epsilonUsagePercent > 90 ? 'text-red-800' : 'text-yellow-800'
            )}>
              Privacy Budget Alert
            </p>
            <p className={clsx(
              'text-sm',
              epsilonUsagePercent > 90 ? 'text-red-600' : 'text-yellow-600'
            )}>
              {epsilonUsagePercent.toFixed(0)}% of privacy budget consumed. 
              {epsilonUsagePercent > 90 ? ' Consider stopping training to preserve differential privacy guarantees.' : ' Monitor closely.'}
            </p>
          </div>
        </div>
      )}
      
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card bg-gradient-to-br from-purple-600 to-purple-700 text-white">
          <div className="flex items-center gap-3 mb-3">
            <Shield className="w-6 h-6" />
            <span className="font-medium">Privacy Budget (ε)</span>
          </div>
          <p className="text-4xl font-bold">{privacyMetrics.epsilonCurrent}</p>
          <p className="text-purple-200 mt-1">of {privacyMetrics.epsilonTotal} total</p>
          <div className="mt-4 bg-purple-500/30 rounded-full h-2">
            <div 
              className="bg-white rounded-full h-2 transition-all"
              style={{ width: `${epsilonUsagePercent}%` }}
            />
          </div>
          <p className="text-purple-200 text-xs mt-2">{(privacyMetrics.epsilonTotal - privacyMetrics.epsilonCurrent).toFixed(1)} remaining</p>
        </div>
        
        <MetricCard
          icon={TrendingDown}
          label="Delta (δ)"
          value="10⁻⁵"
          subtext="Probability of privacy breach"
          color="text-blue-600"
        />
        
        <MetricCard
          icon={Eye}
          label="Noise Multiplier"
          value={privacyMetrics.noiseMultiplier}
          subtext="Gaussian noise scale (σ)"
          color="text-orange-600"
        />
        
        <MetricCard
          icon={EyeOff}
          label="Gradient Clip Norm"
          value={privacyMetrics.clipNorm}
          subtext="Max L2 norm per sample"
          color="text-green-600"
        />
      </div>
      
      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-4">Cumulative Privacy Budget Usage</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={epsilonHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="round" stroke="#64748b" fontSize={12} label={{ value: 'Round', position: 'bottom' }} />
                <YAxis stroke="#64748b" fontSize={12} domain={[0, 10]} />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="cumulative" 
                  stroke="#9333ea" 
                  strokeWidth={2}
                  dot={{ r: 4, fill: '#9333ea' }}
                  name="Cumulative ε"
                />
                <Line 
                  type="monotone" 
                  dataKey="epsilon" 
                  stroke="#c084fc" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ r: 3, fill: '#c084fc' }}
                  name="Per-round ε"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-4">Noise Distribution (Gaussian)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={noiseDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="bin" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip />
                <Bar dataKey="count" fill="#9333ea" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* PII Protection Status */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-slate-800">PII Field Protection</h3>
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm font-medium">{piiFields.length}/{piiFields.length} Protected</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-slate-500 border-b border-slate-200">
                <th className="pb-3 font-medium">Field</th>
                <th className="pb-3 font-medium">Protection Technique</th>
                <th className="pb-3 font-medium text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {piiFields.map((field) => (
                <tr key={field.field} className="border-b border-slate-100 last:border-0">
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <Lock className="w-4 h-4 text-green-500" />
                      <span className="font-medium text-slate-900">{field.field}</span>
                    </div>
                  </td>
                  <td className="py-3 text-slate-600">{field.technique}</td>
                  <td className="py-3 text-center">
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                      <CheckCircle className="w-3 h-3" />
                      Protected
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Audit Log */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <History className="w-5 h-5 text-slate-600" />
          <h3 className="font-semibold text-slate-800">Privacy Audit Log</h3>
        </div>
        <div className="space-y-3">
          {auditLogs.map((log) => (
            <div 
              key={log.id}
              className={clsx(
                'p-3 rounded-lg border flex items-start gap-3',
                log.severity === 'info' && 'bg-blue-50 border-blue-200',
                log.severity === 'warning' && 'bg-yellow-50 border-yellow-200',
                log.severity === 'success' && 'bg-green-50 border-green-200',
              )}
            >
              {log.severity === 'info' && <Info className="w-5 h-5 text-blue-500 mt-0.5" />}
              {log.severity === 'warning' && <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" />}
              {log.severity === 'success' && <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />}
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <p className="font-medium text-slate-900">{log.event}</p>
                  <span className="text-xs text-slate-500 font-mono">{log.timestamp}</span>
                </div>
                <p className="text-sm text-slate-600 mt-1">{log.details}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
