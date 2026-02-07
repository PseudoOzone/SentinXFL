/**
 * Employee - Global Report Generation
 * Generate and view cross-bank intelligence reports.
 */
import { useState, useEffect } from 'react'
import { FileText, Globe, AlertTriangle, Shield, ChevronDown, ChevronUp } from 'lucide-react'
import * as api from '../../api/knowledge'

export default function GlobalReports() {
  const [reports, setReports] = useState<api.ReportData[]>([])
  const [expandedReport, setExpandedReport] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)
  const [, setLoading] = useState(true)

  useEffect(() => { loadReports() }, [])

  const loadReports = async () => {
    setLoading(true)
    try {
      const res = await api.listReports()
      setReports(Array.isArray(res) ? res : [])
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const generate = async (type: 'global' | 'emergent' | 'compliance') => {
    setGenerating(true)
    try {
      let report: api.ReportData
      switch (type) {
        case 'global':
          report = await api.generateGlobalReport()
          break
        case 'emergent':
          report = await api.generateEmergentBriefing()
          break
        case 'compliance':
          report = await api.generateComplianceReport()
          break
      }
      setReports((prev) => [report, ...prev])
      setExpandedReport(report.report_id)
    } catch (e: any) {
      alert(e.message || 'Failed to generate report')
    }
    setGenerating(false)
  }

  const reportIcon = (type: string) => {
    switch (type) {
      case 'emergent': return <AlertTriangle className="w-5 h-5 text-red-500" />
      case 'compliance': return <Shield className="w-5 h-5 text-green-500" />
      case 'global': return <Globe className="w-5 h-5 text-blue-500" />
      default: return <FileText className="w-5 h-5 text-slate-500" />
    }
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
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Globe className="w-7 h-7 text-blue-600" />
            Global Reports
          </h1>
          <p className="text-slate-500 mt-1">Generate cross-bank intelligence and compliance reports</p>
        </div>
      </div>

      {/* Generate Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={() => generate('global')}
          disabled={generating}
          className="bg-white rounded-xl border border-slate-200 p-6 text-left hover:border-blue-400 hover:shadow-md transition-all disabled:opacity-50"
        >
          <Globe className="w-8 h-8 text-blue-500 mb-3" />
          <h3 className="font-semibold text-slate-800">Global Intelligence Report</h3>
          <p className="text-xs text-slate-500 mt-1">Comprehensive cross-bank fraud intelligence with pattern analysis, trends, and bank risk assessments.</p>
        </button>
        <button
          onClick={() => generate('emergent')}
          disabled={generating}
          className="bg-white rounded-xl border border-slate-200 p-6 text-left hover:border-red-400 hover:shadow-md transition-all disabled:opacity-50"
        >
          <AlertTriangle className="w-8 h-8 text-red-500 mb-3" />
          <h3 className="font-semibold text-slate-800">Emergent Attack Briefing</h3>
          <p className="text-xs text-slate-500 mt-1">Focused briefing on zero-day attacks, emergent patterns, and critical alerts requiring immediate action.</p>
        </button>
        <button
          onClick={() => generate('compliance')}
          disabled={generating}
          className="bg-white rounded-xl border border-slate-200 p-6 text-left hover:border-green-400 hover:shadow-md transition-all disabled:opacity-50"
        >
          <Shield className="w-8 h-8 text-green-500 mb-3" />
          <h3 className="font-semibold text-slate-800">Compliance & Audit Report</h3>
          <p className="text-xs text-slate-500 mt-1">Privacy compliance, differential privacy audit trail, and regulatory documentation.</p>
        </button>
      </div>

      {/* Reports List */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">Generated Reports</h2>
        {reports.length > 0 ? reports.map((report) => (
          <div key={report.report_id} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <button
              onClick={() => setExpandedReport(expandedReport === report.report_id ? null : report.report_id)}
              className="w-full p-5 flex items-center justify-between hover:bg-slate-50 transition-colors"
            >
              <div className="flex items-center gap-3 text-left">
                {reportIcon(report.report_type)}
                <div>
                  <p className="font-semibold text-slate-800">{report.title}</p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {new Date(report.generated_at).toLocaleString()} · {report.report_type} · for: {report.generated_for}
                  </p>
                </div>
              </div>
              {expandedReport === report.report_id ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
            </button>

            {expandedReport === report.report_id && (
              <div className="border-t border-slate-100 p-5 space-y-4">
                <div className="bg-slate-50 rounded-lg p-4">
                  <p className="text-sm text-slate-700 font-medium">{report.summary}</p>
                </div>

                {report.sections?.map((section, i) => (
                  <div key={i} className={`border-l-4 rounded-lg p-4 ${severityStyle(section.severity)}`}>
                    <h4 className="font-semibold text-slate-800 mb-1">{section.title}</h4>
                    <p className="text-sm text-slate-600 whitespace-pre-line">{section.content}</p>
                    {section.data && Object.keys(section.data).length > 0 && (
                      <details className="mt-2">
                        <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-700">
                          Raw data ({Object.keys(section.data).length} fields)
                        </summary>
                        <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-x-auto max-h-[300px]">
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
          </div>
        )}
      </div>
    </div>
  )
}
