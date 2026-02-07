/**
 * Client - Reports Page
 * View and generate fraud intelligence reports for the bank.
 */
import { useState, useEffect } from 'react'
import { FileText, ChevronDown, ChevronUp, Shield, ClipboardCheck } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'
import * as api from '../../api/knowledge'

export default function ClientReports() {
  const { user } = useAuth()
  const [reports, setReports] = useState<api.ReportData[]>([])
  const [expandedReport, setExpandedReport] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)
  const [, setLoading] = useState(true)

  useEffect(() => {
    loadReports()
  }, [])

  const loadReports = async () => {
    setLoading(true)
    try {
      const res = await api.listReports({ bank_id: user?.bank_id || undefined })
      setReports(Array.isArray(res) ? res : [])
    } catch (e) {
      console.error('Failed to load reports:', e)
    }
    setLoading(false)
  }

  const generateBankReport = async () => {
    if (!user?.bank_id) return
    setGenerating(true)
    try {
      const report = await api.generateBankReport(user.bank_id)
      setReports((prev) => [report, ...prev])
      setExpandedReport(report.report_id)
    } catch (e: any) {
      alert(e.message || 'Failed to generate report')
    }
    setGenerating(false)
  }

  const generateComplianceReport = async () => {
    setGenerating(true)
    try {
      const report = await api.generateComplianceReport()
      setReports((prev) => [report, ...prev])
      setExpandedReport(report.report_id)
    } catch (e: any) {
      alert(e.message || 'Failed to generate report')
    }
    setGenerating(false)
  }

  const severityStyle = (sev: string) => {
    switch (sev) {
      case 'critical': return 'border-l-red-500 bg-red-50'
      case 'warning': return 'border-l-amber-500 bg-amber-50'
      default: return 'border-l-blue-500 bg-blue-50'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Reports</h1>
          <p className="text-slate-500 mt-1">Fraud intelligence and compliance reports</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={generateBankReport}
            disabled={generating || !user?.bank_id}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium"
          >
            <FileText className="w-4 h-4" />
            Generate Bank Report
          </button>
          <button
            onClick={generateComplianceReport}
            disabled={generating}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-800 disabled:opacity-50 text-sm font-medium"
          >
            <ClipboardCheck className="w-4 h-4" />
            Compliance Report
          </button>
        </div>
      </div>

      {/* Reports List */}
      <div className="space-y-4">
        {reports.length > 0 ? reports.map((report) => (
          <div key={report.report_id} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            {/* Report Header */}
            <button
              onClick={() => setExpandedReport(expandedReport === report.report_id ? null : report.report_id)}
              className="w-full p-5 flex items-center justify-between hover:bg-slate-50 transition-colors"
            >
              <div className="flex items-center gap-3 text-left">
                {report.report_type === 'compliance' ? (
                  <Shield className="w-6 h-6 text-green-500" />
                ) : (
                  <FileText className="w-6 h-6 text-blue-500" />
                )}
                <div>
                  <p className="font-semibold text-slate-800">{report.title}</p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {new Date(report.generated_at).toLocaleString()} · {report.report_type}
                  </p>
                </div>
              </div>
              {expandedReport === report.report_id ? (
                <ChevronUp className="w-5 h-5 text-slate-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-slate-400" />
              )}
            </button>

            {/* Expanded Report Content */}
            {expandedReport === report.report_id && (
              <div className="border-t border-slate-100 p-5 space-y-4">
                <p className="text-sm text-slate-600">{report.summary}</p>

                {report.sections?.map((section, i) => (
                  <div
                    key={i}
                    className={`border-l-4 rounded-lg p-4 ${severityStyle(section.severity)}`}
                  >
                    <h4 className="font-semibold text-slate-800 mb-1">{section.title}</h4>
                    <p className="text-sm text-slate-600 whitespace-pre-line">{section.content}</p>
                    {section.data && Object.keys(section.data).length > 0 && (
                      <details className="mt-2">
                        <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-700">
                          View data ({Object.keys(section.data).length} fields)
                        </summary>
                        <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-x-auto">
                          {JSON.stringify(section.data, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )) : (
          <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
            <FileText className="w-12 h-12 text-slate-300 mx-auto mb-3" />
            <p className="text-slate-500">No reports generated yet</p>
            <p className="text-sm text-slate-400 mt-1">Click "Generate Bank Report" to create your first report</p>
          </div>
        )}
      </div>
    </div>
  )
}
