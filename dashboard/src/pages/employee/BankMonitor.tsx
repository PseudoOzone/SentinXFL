/**
 * Employee - Bank Monitor
 * Monitor all participating banks, their health, risk, and data.
 */
import { useState, useEffect } from 'react'
import {
  Building2,
  RefreshCw,
  UserPlus,
  BarChart3,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import * as api from '../../api/knowledge'

export default function BankMonitor() {
  const [banks, setBanks] = useState<api.BankProfile[]>([])
  const [riskScores, setRiskScores] = useState<Record<string, number>>({})
  const [selectedBank, setSelectedBank] = useState<api.BankProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [showRegister, setShowRegister] = useState(false)
  const [newBankId, setNewBankId] = useState('')
  const [newBankName, setNewBankName] = useState('')

  useEffect(() => { loadData() }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [banksRes, riskRes] = await Promise.all([
        api.getBanks(),
        api.getRiskScores(),
      ])
      setBanks(banksRes.banks)
      setRiskScores(riskRes)
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const handleRegister = async () => {
    if (!newBankId || !newBankName) return
    try {
      await fetch('/api/v1/knowledge/banks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('sentinxfl_token')}`,
        },
        body: JSON.stringify({ bank_id: newBankId, display_name: newBankName }),
      })
      setNewBankId('')
      setNewBankName('')
      setShowRegister(false)
      await loadData()
    } catch (e: any) {
      alert(e.message || 'Failed to register bank')
    }
  }

  const fraudRateData = banks
    .filter((b) => b.avg_fraud_rate > 0)
    .sort((a, b) => b.avg_fraud_rate - a.avg_fraud_rate)
    .slice(0, 10)
    .map((b) => ({
      name: b.display_name.slice(0, 15),
      fraud_rate: +(b.avg_fraud_rate * 100).toFixed(2),
      accuracy: +(b.model_accuracy * 100).toFixed(1),
    }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Building2 className="w-7 h-7 text-blue-600" />
            Bank Monitor
          </h1>
          <p className="text-slate-500 mt-1">{banks.length} participating institutions</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowRegister(!showRegister)}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm font-medium"
          >
            <UserPlus className="w-4 h-4" />
            Register Bank
          </button>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Register Bank Form */}
      {showRegister && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-end gap-3">
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-700 mb-1">Bank ID</label>
            <input
              value={newBankId}
              onChange={(e) => setNewBankId(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm"
              placeholder="bank-xxx-001"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-700 mb-1">Display Name</label>
            <input
              value={newBankName}
              onChange={(e) => setNewBankName(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm"
              placeholder="First National Bank"
            />
          </div>
          <button onClick={handleRegister} className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700">
            Register
          </button>
        </div>
      )}

      {/* Fraud Rate Chart */}
      {fraudRateData.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-500" />
            Bank Fraud Rates
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={fraudRateData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="fraud_rate" fill="#ef4444" name="Fraud Rate %" radius={[4, 4, 0, 0]} />
              <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy %" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Banks Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {banks.length > 0 ? banks.map((bank) => (
          <div
            key={bank.bank_id}
            onClick={() => setSelectedBank(selectedBank?.bank_id === bank.bank_id ? null : bank)}
            className={`bg-white rounded-xl border p-5 cursor-pointer transition-all hover:shadow-md ${
              selectedBank?.bank_id === bank.bank_id ? 'border-blue-500 ring-1 ring-blue-200' : 'border-slate-200'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Building2 className="w-5 h-5 text-blue-500" />
                <h4 className="font-semibold text-slate-800">{bank.display_name}</h4>
              </div>
              <RiskBadge risk={riskScores[bank.bank_id] ?? bank.risk_score} />
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
              <Metric label="Transactions" value={bank.total_transactions.toLocaleString()} />
              <Metric label="Fraud Flagged" value={bank.total_fraud_flagged.toLocaleString()} />
              <Metric label="Fraud Rate" value={`${(bank.avg_fraud_rate * 100).toFixed(2)}%`} />
              <Metric label="Accuracy" value={`${(bank.model_accuracy * 100).toFixed(1)}%`} />
              <Metric label="Rounds" value={bank.rounds_participated.toString()} />
              <Metric label="Last Active" value={bank.last_active ? new Date(bank.last_active).toLocaleDateString() : 'Never'} />
            </div>

            {selectedBank?.bank_id === bank.bank_id && (
              <div className="mt-4 pt-4 border-t border-slate-100 text-xs text-slate-500">
                <p>ID: {bank.bank_id}</p>
                <p>Joined: {new Date(bank.joined_at).toLocaleDateString()}</p>
              </div>
            )}
          </div>
        )) : (
          <div className="col-span-full bg-white rounded-xl border border-slate-200 p-12 text-center text-slate-400">
            No banks registered yet
          </div>
        )}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="font-medium text-slate-800">{value}</p>
    </div>
  )
}

function RiskBadge({ risk }: { risk: number }) {
  const color = risk > 0.7 ? 'bg-red-100 text-red-700' :
                risk > 0.4 ? 'bg-amber-100 text-amber-700' :
                'bg-green-100 text-green-700'
  const label = risk > 0.7 ? 'High Risk' : risk > 0.4 ? 'Medium' : 'Low Risk'
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${color}`}>
      {label}
    </span>
  )
}
