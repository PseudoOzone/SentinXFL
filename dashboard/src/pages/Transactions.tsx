import { useState } from 'react'
import { 
  Search, 
  Filter, 
  Download, 
  Eye,
  ChevronLeft,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react'
import clsx from 'clsx'

// Mock transaction data
const transactions = [
  { id: 'TXN-8847293', date: '2024-01-15 14:23:45', amount: 4582.00, merchant: 'Electronics Store', category: 'Electronics', risk: 0.92, status: 'flagged', pattern: 'Card Testing' },
  { id: 'TXN-8847294', date: '2024-01-15 14:25:12', amount: 125.50, merchant: 'Coffee Shop', category: 'Food & Drink', risk: 0.12, status: 'approved', pattern: null },
  { id: 'TXN-8847295', date: '2024-01-15 14:28:33', amount: 892.50, merchant: 'Online Retailer', category: 'Shopping', risk: 0.78, status: 'review', pattern: 'Velocity Anomaly' },
  { id: 'TXN-8847296', date: '2024-01-15 14:31:08', amount: 12500.00, merchant: 'Wire Transfer', category: 'Transfer', risk: 0.95, status: 'blocked', pattern: 'Geographic Anomaly' },
  { id: 'TXN-8847297', date: '2024-01-15 14:35:44', amount: 45.99, merchant: 'Grocery Store', category: 'Groceries', risk: 0.05, status: 'approved', pattern: null },
  { id: 'TXN-8847298', date: '2024-01-15 14:38:21', amount: 234.99, merchant: 'Gas Station', category: 'Automotive', risk: 0.45, status: 'review', pattern: 'New Device' },
  { id: 'TXN-8847299', date: '2024-01-15 14:42:17', amount: 1875.00, merchant: 'Jewelry Store', category: 'Luxury', risk: 0.72, status: 'flagged', pattern: 'Unusual Amount' },
  { id: 'TXN-8847300', date: '2024-01-15 14:45:56', amount: 89.00, merchant: 'Streaming Service', category: 'Entertainment', risk: 0.08, status: 'approved', pattern: null },
]

function RiskIndicator({ risk }: { risk: number }) {
  const riskLevel = risk >= 0.9 ? 'critical' : risk >= 0.7 ? 'high' : risk >= 0.5 ? 'medium' : 'low'
  const riskPercent = Math.round(risk * 100)
  
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-slate-200 rounded-full overflow-hidden">
        <div 
          className={clsx(
            'h-full rounded-full',
            riskLevel === 'critical' && 'bg-red-500',
            riskLevel === 'high' && 'bg-orange-500',
            riskLevel === 'medium' && 'bg-yellow-500',
            riskLevel === 'low' && 'bg-green-500',
          )}
          style={{ width: `${riskPercent}%` }}
        />
      </div>
      <span className={clsx(
        'text-sm font-medium',
        riskLevel === 'critical' && 'text-red-600',
        riskLevel === 'high' && 'text-orange-600',
        riskLevel === 'medium' && 'text-yellow-600',
        riskLevel === 'low' && 'text-green-600',
      )}>
        {riskPercent}%
      </span>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { icon: typeof CheckCircle, class: string }> = {
    approved: { icon: CheckCircle, class: 'bg-green-100 text-green-800' },
    flagged: { icon: AlertTriangle, class: 'bg-red-100 text-red-800' },
    blocked: { icon: XCircle, class: 'bg-red-100 text-red-800' },
    review: { icon: Eye, class: 'bg-yellow-100 text-yellow-800' },
  }
  
  const { icon: Icon, class: className } = config[status] || config.review
  
  return (
    <span className={clsx('inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium', className)}>
      <Icon className="w-3.5 h-3.5" />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

export default function Transactions() {
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  
  const filteredTransactions = transactions.filter(t => {
    if (statusFilter !== 'all' && t.status !== statusFilter) return false
    if (searchQuery && !t.id.toLowerCase().includes(searchQuery.toLowerCase())) return false
    return true
  })

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Transaction Monitor</h1>
          <p className="text-slate-500 mt-1">Real-time fraud detection analysis</p>
        </div>
        <button className="btn-primary flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export Report
        </button>
      </div>
      
      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search transactions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-field pl-10"
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-slate-400" />
            <select 
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="input-field w-auto"
            >
              <option value="all">All Status</option>
              <option value="approved">Approved</option>
              <option value="flagged">Flagged</option>
              <option value="blocked">Blocked</option>
              <option value="review">Under Review</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Transaction table */}
      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="text-left px-6 py-4 text-sm font-semibold text-slate-600">Transaction ID</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-slate-600">Date & Time</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-slate-600">Merchant</th>
                <th className="text-right px-6 py-4 text-sm font-semibold text-slate-600">Amount</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-slate-600">Risk Score</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-slate-600">Status</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-slate-600">Pattern</th>
                <th className="text-center px-6 py-4 text-sm font-semibold text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredTransactions.map((txn) => (
                <tr key={txn.id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-6 py-4">
                    <span className="font-mono text-sm font-medium text-slate-900">{txn.id}</span>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600">{txn.date}</td>
                  <td className="px-6 py-4">
                    <div>
                      <p className="text-sm font-medium text-slate-900">{txn.merchant}</p>
                      <p className="text-xs text-slate-500">{txn.category}</p>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="font-semibold">${txn.amount.toLocaleString()}</span>
                  </td>
                  <td className="px-6 py-4">
                    <RiskIndicator risk={txn.risk} />
                  </td>
                  <td className="px-6 py-4">
                    <StatusBadge status={txn.status} />
                  </td>
                  <td className="px-6 py-4">
                    {txn.pattern ? (
                      <span className="text-sm text-red-600 font-medium">{txn.pattern}</span>
                    ) : (
                      <span className="text-sm text-slate-400">—</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <button className="p-2 hover:bg-slate-100 rounded-lg text-slate-500 hover:text-primary-600">
                      <Eye className="w-5 h-5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* Pagination */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200">
          <p className="text-sm text-slate-500">
            Showing 1-8 of 847,293 transactions
          </p>
          <div className="flex items-center gap-2">
            <button className="p-2 rounded-lg border border-slate-300 hover:bg-slate-50 disabled:opacity-50">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium">1</span>
            <button className="px-4 py-2 rounded-lg hover:bg-slate-100 text-sm">2</button>
            <button className="px-4 py-2 rounded-lg hover:bg-slate-100 text-sm">3</button>
            <span className="text-slate-400">...</span>
            <button className="px-4 py-2 rounded-lg hover:bg-slate-100 text-sm">1000</button>
            <button className="p-2 rounded-lg border border-slate-300 hover:bg-slate-50">
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
