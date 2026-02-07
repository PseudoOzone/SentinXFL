/**
 * SentinXFL - Authentication Context
 * 
 * Provides auth state to the entire app.
 * Handles login, logout, and token persistence.
 */

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

export interface User {
  user_id: string
  email: string
  role: 'client' | 'employee'
  display_name: string
  bank_id?: string | null
}

interface AuthContextType {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isClient: boolean
  isEmployee: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName: string, role: string, bankId?: string) => Promise<void>
  logout: () => void
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

const API_BASE = '/api/v1'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // Load from localStorage on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('sentinxfl_token')
    const savedUser = localStorage.getItem('sentinxfl_user')
    if (savedToken && savedUser) {
      setToken(savedToken)
      setUser(JSON.parse(savedUser))
    }
    setLoading(false)
  }, [])

  const login = async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Login failed' }))
      throw new Error(err.detail || 'Login failed')
    }
    
    const data = await res.json()
    const userData: User = {
      user_id: data.user_id,
      email: data.email,
      role: data.role,
      display_name: data.display_name,
      bank_id: data.bank_id,
    }
    
    setToken(data.token)
    setUser(userData)
    localStorage.setItem('sentinxfl_token', data.token)
    localStorage.setItem('sentinxfl_user', JSON.stringify(userData))
  }

  const register = async (
    email: string, 
    password: string, 
    displayName: string, 
    role: string,
    bankId?: string,
  ) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, display_name: displayName, role, bank_id: bankId }),
    })
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Registration failed' }))
      throw new Error(err.detail || 'Registration failed')
    }
    
    const data = await res.json()
    const userData: User = {
      user_id: data.user_id,
      email: data.email,
      role: data.role,
      display_name: data.display_name,
      bank_id: data.bank_id,
    }
    
    setToken(data.token)
    setUser(userData)
    localStorage.setItem('sentinxfl_token', data.token)
    localStorage.setItem('sentinxfl_user', JSON.stringify(userData))
  }

  const logout = () => {
    if (token) {
      fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {})
    }
    setToken(null)
    setUser(null)
    localStorage.removeItem('sentinxfl_token')
    localStorage.removeItem('sentinxfl_user')
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isClient: user?.role === 'client',
        isEmployee: user?.role === 'employee',
        login,
        register,
        logout,
        loading,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
