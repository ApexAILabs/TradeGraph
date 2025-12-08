import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'

import { Dashboard } from '@/pages/Dashboard'
import { Landing } from '@/pages/Landing'

import './styles/globals.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/app" element={<Dashboard />} />
          </Routes>
        </div>
      </Router>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          className: 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-lg border border-slate-200 dark:border-slate-800',
        }}
      />
    </QueryClientProvider>
  )
}

export default App
