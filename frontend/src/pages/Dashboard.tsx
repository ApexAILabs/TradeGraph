import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import {
  ArrowTrendingUpIcon,
  ArrowPathIcon,
  ShieldCheckIcon,
  BoltIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

import GraphVisualization from '@/components/GraphVisualization'
import { runDetailedAnalysis, fetchKnowledgeGraphArticles, fetchRecentQueries } from '@/services/api'

interface Recommendation {
  symbol: string
  recommendation: string
  confidence_score: number
  recommended_allocation?: number
  risk_level?: string
  target_price?: number
  stop_loss?: number
  analyst_notes?: string
  key_factors?: string[]
  catalysts?: string[]
}

interface AdvisorReport {
  stance: string
  summary: string
  positions: Array<{
    symbol?: string
    action?: string
    confidence?: number
    allocation?: number
    target_price?: number
    stop_loss?: number
    risk?: string
    should_buy?: boolean
    notes?: string
    catalysts?: string[]
  }>
  risk_summary?: {
    portfolio?: string | null
    position_risk?: string | null
    buy_signals?: number
    sell_signals?: number
  }
  sentiment_overview?: Array<{
    symbol: string
    label?: string
    score?: number
    confidence?: number
    news_summary?: string
    article_count?: number
  }>
  news_highlights?: Array<{
    title?: string
    source?: string
    sentiment?: string
    impact_score?: number
    url?: string
  }>
}

interface AnalysisResult {
  analysis_summary?: {
    symbols_analyzed?: string[]
    portfolio_size?: number
    risk_tolerance?: string
    time_horizon?: string
    asset_breakdown?: Record<string, number>
    symbol_metadata?: Record<string, { resolved_symbol: string; asset_type: string }>
  }
  portfolio_recommendation?: {
    overall_risk_level?: string
    total_confidence?: number
    expected_return?: number
    expected_volatility?: number
    diversification_score?: number
    sector_weights?: Record<string, number>
    recommendations?: Recommendation[]
  }
  sentiment_analysis?: Record<string, any>
  news_data?: { articles?: any[]; total_count?: number }
  recommendations?: Recommendation[]
  advisor_report?: AdvisorReport | null
  database_insights?: {
    price_history?: Record<string, Array<{ date: string; close: number }>>
    knowledge_graph_articles?: any[]
    recent_queries?: any[]
  }
}

const defaultSymbols = 'AAPL, BTC'

export function Dashboard() {
  const [symbolsInput, setSymbolsInput] = useState(defaultSymbols)
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [dbArticles, setDbArticles] = useState<any[]>([])
  const [dbQueries, setDbQueries] = useState<any[]>([])

  const analysisMutation = useMutation({
    mutationFn: runDetailedAnalysis,
    onSuccess: async (data: AnalysisResult) => {
      const firstSymbol = data?.analysis_summary?.symbols_analyzed?.[0]
      setSelectedSymbol(firstSymbol ?? null)

      try {
        const symbolsForDb = data?.analysis_summary?.symbols_analyzed ?? []
        const [articles, queries] = await Promise.all([
          fetchKnowledgeGraphArticles(symbolsForDb),
          fetchRecentQueries(5),
        ])
        setDbArticles(articles)
        setDbQueries(queries)
      } catch (err) {
        console.error(err)
      }
    },
    onError: () => {
      toast.error('Unable to complete the analysis. Please try again in a moment.')
    },
  })

  const analysis = analysisMutation.data as AnalysisResult | undefined

  useEffect(() => {
    if (analysis?.analysis_summary?.symbols_analyzed?.length && !selectedSymbol) {
      setSelectedSymbol(analysis.analysis_summary.symbols_analyzed[0])
    }
  }, [analysis, selectedSymbol])

  const priceSeries = useMemo(() => {
    if (!analysis?.database_insights?.price_history || !selectedSymbol) {
      return []
    }
    const history =
      analysis.database_insights.price_history[selectedSymbol] ??
      analysis.database_insights.price_history[
        analysis.analysis_summary?.symbol_metadata?.[selectedSymbol]?.resolved_symbol ?? ''
      ]
    if (!history) {
      return []
    }
    return history.map((point) => ({
      date: new Date(point.date).toLocaleDateString(),
      close: point.close,
    }))
  }, [analysis, selectedSymbol])

  const expectedReturnValue =
    typeof analysis?.portfolio_recommendation?.expected_return === 'number'
      ? `${((analysis.portfolio_recommendation.expected_return ?? 0) * 100).toFixed(1)}%`
      : '—'

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const symbols = symbolsInput
      .split(',')
      .map((symbol) => symbol.trim().toUpperCase())
      .filter(Boolean)

    if (!symbols.length) {
      toast.error('Add at least one stock or crypto symbol to analyze.')
      return
    }

    analysisMutation.mutate({ symbols, analysis_type: 'detailed' })
  }

  const stanceColor = analysis?.advisor_report?.stance === 'bullish'
    ? 'text-emerald-400'
    : analysis?.advisor_report?.stance === 'defensive'
    ? 'text-amber-400'
    : 'text-blue-400'

  return (
    <div className="min-h-screen bg-slate-900 text-slate-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
        <section className="bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 rounded-3xl p-8 relative overflow-hidden">
          <div className="absolute inset-0 opacity-30">
            <div className="absolute inset-y-0 right-0 w-1/2 bg-[radial-gradient(circle_at_top,_theme(colors.white/20),_transparent_65%)]" />
          </div>
          <div className="relative z-10 space-y-6">
            <div>
              <p className="text-sm uppercase tracking-widest text-blue-100">TradeGraph Intelligence</p>
              <h1 className="text-3xl sm:text-4xl font-bold mt-2">
                AI-native terminal for stocks and crypto assets
              </h1>
              <p className="text-blue-100 mt-2 max-w-2xl">
                Multi-agent workflows combine live market data, DuckDB price archives, and the Neo4j knowledge
                graph to deliver a research-grade opinion on any ticker or token.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="bg-white/10 backdrop-blur rounded-2xl p-4 flex flex-col md:flex-row gap-4">
              <label className="flex-1">
                <span className="text-xs uppercase tracking-widest text-blue-100">Symbols</span>
                <input
                  type="text"
                  value={symbolsInput}
                  onChange={(event) => setSymbolsInput(event.target.value)}
                  placeholder="e.g. AAPL, TSLA, BTC, SOL"
                  className="mt-2 w-full rounded-xl border border-white/20 bg-white/10 px-4 py-3 text-white placeholder:text-blue-100 focus:outline-none focus:ring-2 focus:ring-white/80"
                />
              </label>
              <button
                type="submit"
                disabled={analysisMutation.isPending}
                className="flex items-center justify-center rounded-xl bg-white text-blue-700 font-semibold px-6 py-3 shadow-lg disabled:opacity-60"
              >
                {analysisMutation.isPending ? (
                  <>
                    <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
                    Running analysis
                  </>
                ) : (
                  <>
                    <BoltIcon className="w-5 h-5 mr-2" />
                    Run AI report
                  </>
                )}
              </button>
            </form>

            <div className="flex flex-wrap gap-3 text-xs text-blue-100">
              {['LLM agents', 'DuckDB pricing', 'Neo4j graph', 'Live sentiment', 'SEC/News scraping'].map((item) => (
                <span key={item} className="px-3 py-1 rounded-full bg-white/10 border border-white/20">
                  {item}
                </span>
              ))}
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <InsightCard
            title="Portfolio risk"
            icon={<ShieldCheckIcon className="w-5 h-5" />}
            value={analysis?.portfolio_recommendation?.overall_risk_level || '—'}
            helper={analysis?.analysis_summary?.asset_breakdown
              ? `${analysis.analysis_summary.asset_breakdown.equity ?? 0} equities · ${analysis.analysis_summary.asset_breakdown.crypto ?? 0} crypto`
              : 'Blending equity + crypto'}
          />
          <InsightCard
            title="Expected return"
            icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
            value={expectedReturnValue}
            helper="AI-optimized projection"
          />
          <InsightCard
            title="Sentiment coverage"
            icon={<CheckCircleIcon className="w-5 h-5" />}
            value={`${analysis?.news_data?.total_count ?? 0} articles`}
            helper="Agent + knowledge graph"
          />
          <InsightCard
            title="Database signals"
            icon={<ExclamationTriangleIcon className="w-5 h-5" />}
            value={`${dbArticles.length} KG alerts`}
            helper="Neo4j articles & DuckDB logs"
          />
        </section>

        {analysis ? (
          <div className="space-y-8">
            <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-slate-800 border border-slate-700 rounded-2xl p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-400">Advisor verdict</p>
                    <p className={`text-2xl font-semibold ${stanceColor}`}>
                      {analysis.advisor_report?.stance?.toUpperCase() || 'WAITING'}
                    </p>
                  </div>
                  <span className="text-xs uppercase tracking-widest text-slate-400">
                    {analysis.analysis_summary?.risk_tolerance} · {analysis.analysis_summary?.time_horizon}
                  </span>
                </div>
                <p className="mt-4 text-slate-200">{analysis.advisor_report?.summary}</p>

                <div className="mt-6 space-y-3 max-h-72 overflow-y-auto pr-1">
                  {analysis.advisor_report?.positions?.map((position) => (
                    <div
                      key={`${position.symbol}-${position.action}`}
                      className="bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 flex flex-wrap items-center gap-4"
                    >
                      <div className="flex-1 min-w-[120px]">
                        <p className="font-semibold">{position.symbol}</p>
                        <p className="text-xs text-slate-400">Risk · {position.risk || 'n/a'}</p>
                      </div>
                      <div>
                        <p className={`text-sm font-semibold ${position.should_buy ? 'text-emerald-400' : 'text-amber-400'}`}>
                          {position.action}
                        </p>
                        <p className="text-xs text-slate-400">
                          Confidence {(position.confidence ?? 0) * 100 >= 1 ? `${Math.round((position.confidence || 0) * 100)}%` : 'n/a'}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-400">Allocation</p>
                        <p className="font-semibold">{position.allocation ? `${Math.round(position.allocation * 100)}%` : 'n/a'}</p>
                      </div>
                      {position.notes && (
                        <p className="text-xs text-slate-400 flex-1 min-w-[160px]">
                          {position.notes}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5">
                  <p className="text-sm text-slate-400">Risk summary</p>
                  <p className="text-xl font-bold">
                    {analysis.advisor_report?.risk_summary?.position_risk?.toUpperCase() || 'UNSET'}
                  </p>
                  <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-slate-900/60 rounded-xl p-3">
                      <p className="text-slate-400 text-xs">Buy signals</p>
                      <p className="text-lg font-semibold text-emerald-400">
                        {analysis.advisor_report?.risk_summary?.buy_signals ?? 0}
                      </p>
                    </div>
                    <div className="bg-slate-900/60 rounded-xl p-3">
                      <p className="text-slate-400 text-xs">Sell signals</p>
                      <p className="text-lg font-semibold text-rose-400">
                        {analysis.advisor_report?.risk_summary?.sell_signals ?? 0}
                      </p>
                    </div>
                    <div className="bg-slate-900/60 rounded-xl p-3">
                      <p className="text-slate-400 text-xs">Portfolio risk</p>
                      <p className="text-lg font-semibold">{analysis.portfolio_recommendation?.overall_risk_level || 'n/a'}</p>
                    </div>
                    <div className="bg-slate-900/60 rounded-xl p-3">
                      <p className="text-slate-400 text-xs">Source coverage</p>
                      <p className="text-lg font-semibold">{analysis.news_data?.total_count ?? 0} articles</p>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5">
                  <p className="text-sm text-slate-400">Knowledge graph alerts</p>
                  <div className="mt-3 space-y-3 max-h-48 overflow-y-auto">
                    {dbArticles.length === 0 && <p className="text-sm text-slate-400">Run an analysis to sync Neo4j news.</p>}
                    {dbArticles.map((article) => (
                      <div key={article.id || article.title} className="bg-slate-900/60 rounded-xl p-3">
                        <p className="font-semibold text-sm">{article.title}</p>
                        <p className="text-xs text-slate-400 flex justify-between">
                          <span>{article.source}</span>
                          <span>{article.published_at?.split('T')[0]}</span>
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className="text-sm text-slate-400">Price analytics · DuckDB</p>
                    <select
                      value={selectedSymbol ?? ''}
                      onChange={(event) => setSelectedSymbol(event.target.value)}
                      className="mt-1 bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-1 text-sm"
                    >
                      {(analysis.analysis_summary?.symbols_analyzed ?? []).map((symbol) => (
                        <option key={symbol} value={symbol}>
                          {symbol}
                        </option>
                      ))}
                    </select>
                  </div>
                  <span className="text-xs text-slate-400">{priceSeries.length} stored points</span>
                </div>
                {priceSeries.length ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={priceSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" stroke="#cbd5f5" hide={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                        <YAxis stroke="#cbd5f5" tick={{ fill: '#94a3b8', fontSize: 12 }} domain={['auto', 'auto']} />
                        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }} />
                        <Line type="monotone" dataKey="close" stroke="#38bdf8" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">Run an analysis to populate DuckDB history.</p>
                )}
              </div>

              <div className="bg-slate-800 border border-slate-700 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-2 px-2">
                  <p className="text-sm text-slate-400">Knowledge graph snapshot</p>
                  <span className="text-xs text-slate-400">Neo4j · 100 edges max</span>
                </div>
                <GraphVisualization />
              </div>
            </section>

            <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-slate-800 border border-slate-700 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <p className="text-sm text-slate-400">Live agent news feed</p>
                  <span className="text-xs text-slate-400">{analysis.news_data?.total_count ?? 0} sources</span>
                </div>
                <div className="space-y-4 max-h-96 overflow-y-auto pr-1">
                  {(analysis.news_data?.articles ?? []).slice(0, 12).map((article) => (
                    <motion.a
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                      key={`${article.title}-${article.url}`}
                      href={article.url}
                      target="_blank"
                      rel="noreferrer"
                      className="block bg-slate-900/60 border border-slate-700 rounded-xl p-4 hover:border-blue-400"
                    >
                      <p className="font-semibold">{article.title}</p>
                      <p className="text-xs text-slate-400 mt-1">{article.source}</p>
                      <p className="text-sm text-slate-300 mt-2 line-clamp-2">{article.summary || article.content}</p>
                    </motion.a>
                  ))}
                  {!analysis.news_data?.articles?.length && (
                    <p className="text-sm text-slate-400">News archive will appear after running an analysis.</p>
                  )}
                </div>
              </div>

              <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 space-y-4">
                <div>
                  <p className="text-sm text-slate-400">DuckDB query log</p>
                  <div className="mt-3 space-y-3 max-h-40 overflow-y-auto">
                    {dbQueries.map((query, index) => (
                      <div key={`${query.timestamp}-${index}`} className="bg-slate-900/60 rounded-xl p-3">
                        <p className="text-xs text-slate-400">{new Date(query.timestamp).toLocaleString()}</p>
                        <p className="text-sm font-semibold">{query.query_text}</p>
                      </div>
                    ))}
                    {!dbQueries.length && <p className="text-xs text-slate-400">New analyses will be logged here.</p>}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-slate-400">Sentiment overview</p>
                  <div className="mt-3 space-y-3">
                    {analysis.advisor_report?.sentiment_overview?.map((sentiment) => (
                      <div key={sentiment.symbol} className="bg-slate-900/60 border border-slate-700 rounded-xl p-3">
                        <p className="font-semibold">{sentiment.symbol}</p>
                        <p className="text-xs text-slate-400">
                          {sentiment.label} · score {(sentiment.score ?? 0).toFixed(2)} · {sentiment.article_count} articles
                        </p>
                        <p className="text-xs text-slate-300 mt-1 line-clamp-2">{sentiment.news_summary}</p>
                      </div>
                    ))}
                    {!analysis.advisor_report?.sentiment_overview?.length && (
                      <p className="text-xs text-slate-400">Sentiment summaries will populate after the first run.</p>
                    )}
                  </div>
                </div>
              </div>
            </section>
          </div>
        ) : (
          <div className="bg-slate-800 border border-slate-700 rounded-2xl p-10 text-center">
            <p className="text-lg font-semibold">Ready when you are</p>
            <p className="text-slate-400 mt-2">
              Add tickers above to generate a full report with market data, knowledge graph news, and multi-agent advice.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function InsightCard({
  title,
  icon,
  value,
  helper,
}: {
  title: string
  icon: ReactNode
  value: string
  helper: string
}) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5">
      <div className="flex items-center gap-3 text-slate-200">
        <span className="w-10 h-10 rounded-full bg-slate-900/60 flex items-center justify-center text-blue-300">
          {icon}
        </span>
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-400">{title}</p>
          <p className="text-xl font-semibold">{value}</p>
        </div>
      </div>
      <p className="mt-3 text-xs text-slate-400">{helper}</p>
    </div>
  )
}
