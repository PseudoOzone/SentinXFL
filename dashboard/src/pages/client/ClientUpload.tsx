/**
 * Client - File Upload Page
 * Upload CSV/Parquet files for fraud analysis.
 */
import { useState, useEffect, useRef } from 'react'
import { Upload, FileText, CheckCircle, XCircle, Loader2, Trash2 } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'
import * as api from '../../api/knowledge'

export default function ClientUpload() {
  const { user } = useAuth()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploads, setUploads] = useState<api.UploadInfo[]>([])
  const [uploading, setUploading] = useState(false)
  const [description, setDescription] = useState('')
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)

  useEffect(() => {
    loadUploads()
  }, [])

  const loadUploads = async () => {
    try {
      const res = await api.listUploads({ bank_id: user?.bank_id || undefined })
      setUploads(res.uploads)
    } catch (e) {
      console.error('Failed to load uploads:', e)
    }
  }

  const handleUpload = async (file: File) => {
    if (!user?.bank_id) {
      setError('No bank ID associated with your account')
      return
    }

    setUploading(true)
    setError('')
    try {
      await api.uploadFile(file, user.bank_id, description)
      setDescription('')
      await loadUploads()
    } catch (e: any) {
      setError(e.message || 'Upload failed')
    }
    setUploading(false)
  }

  const handleProcess = async (uploadId: string) => {
    try {
      await api.processUpload(uploadId)
      await loadUploads()
    } catch (e: any) {
      setError(e.message || 'Processing failed')
    }
  }

  const handleDelete = async (uploadId: string) => {
    if (!confirm('Delete this upload?')) return
    try {
      await api.deleteUpload(uploadId)
      await loadUploads()
    } catch (e: any) {
      setError(e.message || 'Delete failed')
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files?.[0]) {
      handleUpload(e.dataTransfer.files[0])
    }
  }

  const statusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'error': return <XCircle className="w-5 h-5 text-red-500" />
      case 'processing': return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      default: return <FileText className="w-5 h-5 text-slate-400" />
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Upload Data</h1>
        <p className="text-slate-500 mt-1">Upload transaction data for fraud analysis</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm">
          {error}
        </div>
      )}

      {/* Upload Zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors ${
          dragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-slate-50 hover:border-blue-400'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
        <p className="text-lg font-medium text-slate-700 mb-1">
          Drop your CSV or Parquet file here
        </p>
        <p className="text-sm text-slate-500 mb-4">or click to browse (max 500MB)</p>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.parquet"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
        />

        <div className="flex items-center gap-3 justify-center">
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional description..."
            className="px-3 py-2 border border-slate-300 rounded-lg text-sm w-64"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium text-sm"
          >
            {uploading ? 'Uploading...' : 'Browse Files'}
          </button>
        </div>
      </div>

      {/* Upload History */}
      <div className="bg-white rounded-xl border border-slate-200">
        <div className="p-5 border-b border-slate-100">
          <h3 className="text-lg font-semibold text-slate-800">Upload History</h3>
        </div>
        <div className="divide-y divide-slate-100">
          {uploads.length > 0 ? uploads.map((u) => (
            <div key={u.upload_id} className="p-4 flex items-center gap-4">
              {statusIcon(u.status)}
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm text-slate-800 truncate">{u.filename}</p>
                <p className="text-xs text-slate-500 mt-0.5">
                  {(u.file_size / 1024 / 1024).toFixed(1)}MB
                  {u.row_count != null && ` · ${u.row_count.toLocaleString()} rows`}
                  {u.fraud_count != null && ` · ${u.fraud_count.toLocaleString()} fraud`}
                  {` · ${new Date(u.uploaded_at).toLocaleDateString()}`}
                </p>
                {u.error_message && (
                  <p className="text-xs text-red-500 mt-0.5">{u.error_message}</p>
                )}
              </div>
              <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                u.status === 'completed' ? 'bg-green-100 text-green-700' :
                u.status === 'error' ? 'bg-red-100 text-red-700' :
                u.status === 'processing' ? 'bg-blue-100 text-blue-700' :
                'bg-slate-100 text-slate-600'
              }`}>
                {u.status}
              </span>
              {u.status === 'uploaded' && (
                <button
                  onClick={() => handleProcess(u.upload_id)}
                  className="px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium hover:bg-blue-700"
                >
                  Process
                </button>
              )}
              <button
                onClick={() => handleDelete(u.upload_id)}
                className="p-2 text-slate-400 hover:text-red-500 rounded-lg hover:bg-slate-100"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )) : (
            <div className="p-8 text-center text-slate-400">No uploads yet</div>
          )}
        </div>
      </div>
    </div>
  )
}
