/**
 * Employee - Pattern Management
 * View, search, and manage all fraud patterns in the library.
 */
import { useState, useEffect } from 'react'
import { Search, Layers, ShieldCheck, Zap, AlertTriangle, Eye } from 'lucide-react'
import * as api from '../../api/knowledge'

const typeIcons: Record<string, any> = {
  fact: ShieldCheck,
  emergent: Zap,
  variant: Layers,
  zero_day: AlertTriangle,
  historical: Eye,
}

const typeColors: Record<string, string> = {
  fact: 'bg-green-100 text-green-700',
  emergent: 'bg-amber-100 text-amber-700',
  variant: 'bg-blue-100 text-blue-700',
  zero_day: 'bg-red-100 text-red-700',
  historical: 'bg-slate-100 text-slate-600',
}

const severityColors: Record<string, string> = {
  low: 'bg-green-100 text-green-700',
  medium: 'bg-yellow-100 text-yellow-700',
  high: 'bg-orange-100 text-orange-700',
  critical: 'bg-red-100 text-red-700',
}

export default function PatternManagement() {
  const [patterns, setPatterns] = useState<api.PatternEntry[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [sevFilter, setSevFilter] = useState<string>('')
  const [selectedPattern, setSelectedPattern] = useState<api.PatternEntry | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<Record<string, unknown>>({})

  useEffect(() => { loadPatterns() }, [typeFilter, sevFilter])

  const loadPatterns = async () => {
    setLoading(true)
    try {
      const [pRes, sRes] = await Promise.all([
        api.getPatterns({ pattern_type: typeFilter || undefined, severity: sevFilter || undefined, limit: 200 }),
        api.getLibraryStatistics(),
      ])
      setPatterns(pRes.patterns)
      setStats(sRes)
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) { loadPatterns(); return }
    setLoading(true)
    try {
      const res = await api.searchPatterns(searchQuery)
      setPatterns(res.patterns)
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const totalPatterns = (stats as any)?.total_patterns ?? patterns.length

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <Layers className="w-7 h-7 text-purple-600" />
          Pattern Library
        </h1>
        <p className="text-slate-500 mt-1">{totalPatterns} patterns tracked across all institutions</p>
      </div>

      {/* Search + Filters */}
      <div className="flex flex-wrap gap-3">
        <div className="flex-1 min-w-[300px] relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search patterns..."
            className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm"
          />
        </div>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="px-3 py-2.5 border border-slate-300 rounded-lg text-sm"
        >
          <option value="">All Types</option>
          <option value="fact">Fact-Based</option>
          <option value="emergent">Emergent</option>
          <option value="variant">Variant</option>
          <option value="zero_day">Zero-Day</option>
          <option value="historical">Historical</option>
        </select>
        <select
          value={sevFilter}
          onChange={(e) => setSevFilter(e.target.value)}
          className="px-3 py-2.5 border border-slate-300 rounded-lg text-sm"
        >
          <option value="">All Severities</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="critical">Critical</option>
        </select>
        <button onClick={handleSearch} className="px-4 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700">
          Search
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pattern List */}
        <div className="lg:col-span-2 space-y-3">
          {loading ? (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center text-slate-400">Loading patterns...</div>
          ) : patterns.length > 0 ? patterns.map((p) => {
            const TypeIcon = typeIcons[p.pattern_type] || Layers
            return (
              <div
                key={p.pattern_id}
                onClick={() => setSelectedPattern(p)}
                className={`bg-white rounded-xl border p-4 cursor-pointer transition-all hover:shadow-md ${
                  selectedPattern?.pattern_id === p.pattern_id ? 'border-blue-500 ring-1 ring-blue-200' : 'border-slate-200'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg ${typeColors[p.pattern_type] || 'bg-slate-100'}`}>
                    <TypeIcon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="font-semibold text-sm text-slate-800 truncate">{p.title}</p>
                      <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${severityColors[p.severity] || ''}`}>
                        {p.severity}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 line-clamp-2">{p.description}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-slate-400">
                      <span>Confidence: {(p.confidence * 100).toFixed(0)}%</span>
                      <span>Banks: {p.source_bank_count}</span>
                      <span>Obs: {p.observation_count}</span>
                      <span>Novelty: {(p.novelty_score * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )
          }) : (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center text-slate-400">
              No patterns match your filters
            </div>
          )}
        </div>

        {/* Pattern Detail */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 sticky top-6 self-start">
          {selectedPattern ? (
            <div className="space-y-4">
              <div>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${typeColors[selectedPattern.pattern_type]}`}>
                  {selectedPattern.pattern_type}
                </span>
                <h3 className="text-lg font-bold text-slate-800 mt-2">{selectedPattern.title}</h3>
              </div>
              <p className="text-sm text-slate-600">{selectedPattern.description}</p>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <Detail label="Severity" value={selectedPattern.severity} />
                <Detail label="Confidence" value={`${(selectedPattern.confidence * 100).toFixed(1)}%`} />
                <Detail label="Banks" value={selectedPattern.source_bank_count.toString()} />
                <Detail label="Observations" value={selectedPattern.observation_count.toString()} />
                <Detail label="Novelty" value={`${(selectedPattern.novelty_score * 100).toFixed(1)}%`} />
                <Detail label="Source" value={selectedPattern.source} />
              </div>

              <div>
                <p className="text-xs font-medium text-slate-500 mb-1">First Seen</p>
                <p className="text-sm text-slate-700">{new Date(selectedPattern.first_seen).toLocaleString()}</p>
              </div>

              {selectedPattern.indicators && Object.keys(selectedPattern.indicators).length > 0 && (
                <div>
                  <p className="text-xs font-medium text-slate-500 mb-1">Indicators</p>
                  <div className="bg-slate-50 rounded-lg p-3 text-xs font-mono">
                    {Object.entries(selectedPattern.indicators).map(([k, v]) => (
                      <div key={k}>{k}: {typeof v === 'number' ? v.toFixed(6) : v}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-400">
              <Layers className="w-10 h-10 mx-auto mb-3" />
              <p>Select a pattern to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="font-medium text-slate-800 capitalize">{value}</p>
    </div>
  )
}
