import { useState } from 'react'
import {
  Brain,
  Search,
  Send,
  Lightbulb,
  TrendingUp,
  Info,
  ChevronRight
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'
import clsx from 'clsx'

// Mock transaction for explanation
const sampleTransaction = {
  id: 'TXN_2025_00847',
  amount: 15750.00,
  merchant: 'Overseas Electronics Ltd',
  category: 'Electronics',
  location: 'Hong Kong',
  time: '03:24 AM',
  riskScore: 0.87,
  riskLevel: 'high'
}

// Feature importance data
const featureImportance = [
  { feature: 'Transaction Amount', importance: 0.32, direction: 'positive' },
  { feature: 'Time of Day', importance: 0.24, direction: 'positive' },
  { feature: 'Location Risk', importance: 0.18, direction: 'positive' },
  { feature: 'Merchant Category', importance: 0.12, direction: 'positive' },
  { feature: 'Account Age', importance: -0.08, direction: 'negative' },
  { feature: 'Transaction Frequency', importance: 0.06, direction: 'positive' },
]

// Chat messages
const initialMessages = [
  {
    role: 'assistant',
    content: 'Hello! I\'m the SentinXFL AI Assistant. I can help you understand fraud detection results, explain model decisions, and answer questions about suspicious transactions. What would you like to know?'
  }
]

// Sample explanations
const sampleExplanations = [
  'The transaction amount of $15,750 is significantly higher than typical purchases in this category.',
  'The transaction occurred at 3:24 AM, which is outside normal banking hours for this account.',
  'The merchant location (Hong Kong) represents a high-risk jurisdiction.',
  'This is the first transaction with this merchant for this account.'
]

function FeatureBar({ feature, importance, direction }: { feature: string, importance: number, direction: string }) {
  const absImportance = Math.abs(importance)
  const percentage = absImportance * 100
  
  return (
    <div className="flex items-center gap-4">
      <div className="w-40 text-sm text-slate-700 truncate">{feature}</div>
      <div className="flex-1 flex items-center gap-2">
        <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
          <div 
            className={clsx(
              'h-full rounded-full transition-all',
              direction === 'positive' ? 'bg-red-500' : 'bg-green-500'
            )}
            style={{ width: `${percentage * 3}%` }}
          />
        </div>
        <span className={clsx(
          'text-sm font-mono w-16 text-right',
          direction === 'positive' ? 'text-red-600' : 'text-green-600'
        )}>
          {direction === 'positive' ? '+' : ''}{(importance * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

export default function Explainability() {
  const [messages, setMessages] = useState(initialMessages)
  const [inputMessage, setInputMessage] = useState('')
  const [selectedTxnId, setSelectedTxnId] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  
  const handleSendMessage = () => {
    if (!inputMessage.trim()) return
    
    setMessages(prev => [...prev, { role: 'user', content: inputMessage }])
    setIsLoading(true)
    
    // Simulate AI response
    setTimeout(() => {
      const response = inputMessage.toLowerCase().includes('amount') 
        ? 'The transaction amount of $15,750 is 847% higher than the average for this account ($1,858). Large, unusual amounts are a key indicator of potential fraud, especially when combined with other risk factors.'
        : inputMessage.toLowerCase().includes('time')
        ? 'The transaction occurred at 3:24 AM local time. Analysis shows 78% of fraudulent transactions for this account type occur between 1-5 AM, compared to only 12% of legitimate transactions.'
        : 'Based on the federated learning model trained across 6 participating banks, this transaction exhibits multiple high-risk characteristics. The combination of factors results in an 87% fraud probability score.'
      
      setMessages(prev => [...prev, { role: 'assistant', content: response }])
      setIsLoading(false)
    }, 1000)
    
    setInputMessage('')
  }
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Explainability Center</h1>
          <p className="text-slate-500 mt-1">AI-powered fraud explanation & insights</p>
        </div>
      </div>
      
      {/* Transaction Search */}
      <div className="card">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Enter Transaction ID to analyze (e.g., TXN_2025_00847)"
              value={selectedTxnId}
              onChange={(e) => setSelectedTxnId(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <button className="btn-primary">
            <Brain className="w-5 h-5 mr-2" />
            Analyze
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Transaction Details & Explanation */}
        <div className="space-y-6">
          {/* Transaction Summary */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-slate-800">Transaction Analysis</h3>
              <span className={clsx(
                'px-3 py-1 rounded-full text-sm font-medium',
                sampleTransaction.riskLevel === 'high' && 'bg-red-100 text-red-700',
                sampleTransaction.riskLevel === 'medium' && 'bg-yellow-100 text-yellow-700',
                sampleTransaction.riskLevel === 'low' && 'bg-green-100 text-green-700',
              )}>
                {(sampleTransaction.riskScore * 100).toFixed(0)}% Risk Score
              </span>
            </div>
            
            <div className="bg-slate-50 rounded-lg p-4 mb-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-slate-500">Transaction ID</p>
                  <p className="font-mono text-sm text-slate-900">{sampleTransaction.id}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Amount</p>
                  <p className="font-bold text-lg text-slate-900">${sampleTransaction.amount.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Merchant</p>
                  <p className="text-slate-900">{sampleTransaction.merchant}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Location</p>
                  <p className="text-slate-900">{sampleTransaction.location}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Category</p>
                  <p className="text-slate-900">{sampleTransaction.category}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Time</p>
                  <p className="text-slate-900">{sampleTransaction.time}</p>
                </div>
              </div>
            </div>
            
            {/* Key Findings */}
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-500" />
                Key Findings
              </h4>
              <div className="space-y-2">
                {sampleExplanations.map((explanation, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <ChevronRight className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                    <span className="text-slate-700">{explanation}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Feature Importance */}
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-primary-600" />
              <h3 className="font-semibold text-slate-800">Feature Importance (SHAP)</h3>
            </div>
            
            <div className="space-y-3">
              {featureImportance.map((item) => (
                <FeatureBar 
                  key={item.feature}
                  feature={item.feature}
                  importance={item.importance}
                  direction={item.direction}
                />
              ))}
            </div>
            
            <div className="mt-4 pt-4 border-t border-slate-200 flex items-start gap-2">
              <Info className="w-4 h-4 text-blue-500 mt-0.5" />
              <p className="text-xs text-slate-500">
                <span className="text-red-600 font-medium">Red bars</span> indicate features that increase fraud probability, 
                while <span className="text-green-600 font-medium">green bars</span> indicate features that decrease it.
              </p>
            </div>
          </div>
          
          {/* Feature Importance Chart */}
          <div className="card">
            <h3 className="font-semibold text-slate-800 mb-4">Feature Impact Visualization</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                  data={featureImportance} 
                  layout="vertical"
                  margin={{ left: 100 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" domain={[-0.15, 0.4]} stroke="#64748b" fontSize={12} />
                  <YAxis type="category" dataKey="feature" stroke="#64748b" fontSize={11} width={100} />
                  <Tooltip />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {featureImportance.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.direction === 'positive' ? '#ef4444' : '#22c55e'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        
        {/* AI Chat Interface */}
        <div className="card flex flex-col h-[calc(100vh-200px)] min-h-[600px]">
          <div className="flex items-center gap-3 pb-4 border-b border-slate-200">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-800">SentinXFL AI Assistant</h3>
              <p className="text-xs text-green-600">● Online</p>
            </div>
          </div>
          
          {/* Messages */}
          <div className="flex-1 overflow-y-auto py-4 space-y-4">
            {messages.map((message, idx) => (
              <div 
                key={idx}
                className={clsx(
                  'flex',
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                <div className={clsx(
                  'max-w-[80%] p-3 rounded-lg',
                  message.role === 'user' 
                    ? 'bg-primary-600 text-white rounded-br-none'
                    : 'bg-slate-100 text-slate-800 rounded-bl-none'
                )}>
                  <p className="text-sm">{message.content}</p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-100 p-3 rounded-lg rounded-bl-none">
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Suggested Questions */}
          <div className="py-3 border-t border-slate-200">
            <p className="text-xs text-slate-500 mb-2">Suggested questions:</p>
            <div className="flex flex-wrap gap-2">
              {['Why is the amount suspicious?', 'Explain the time factor', 'What patterns indicate fraud?'].map((q) => (
                <button
                  key={q}
                  onClick={() => setInputMessage(q)}
                  className="text-xs px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-full text-slate-700 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
          
          {/* Input */}
          <div className="pt-3 border-t border-slate-200">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Ask about this transaction..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
              <button 
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
