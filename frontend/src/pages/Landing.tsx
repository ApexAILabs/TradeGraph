import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ChartBarIcon,
  CpuChipIcon,
  ShieldCheckIcon,
  BoltIcon,
  ArrowRightIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

const features = [
  {
    name: 'AI-Powered Analysis',
    description: 'Multi-agent AI system that analyzes news, technical indicators, and financial reports to provide comprehensive insights.',
    icon: CpuChipIcon,
    gradient: 'from-blue-500 to-purple-600',
  },
  {
    name: 'Real-time Market Data',
    description: 'Live market data, technical analysis, and sentiment tracking with instant WebSocket updates.',
    icon: BoltIcon,
    gradient: 'from-yellow-500 to-orange-600',
  },
  {
    name: 'Smart Recommendations',
    description: 'Intelligent buy/sell/hold recommendations with confidence scores, target prices, and risk assessment.',
    icon: ChartBarIcon,
    gradient: 'from-green-500 to-teal-600',
  },
  {
    name: 'Risk Management',
    description: 'Advanced portfolio optimization, risk assessment, and position sizing based on your risk tolerance.',
    icon: ShieldCheckIcon,
    gradient: 'from-red-500 to-pink-600',
  },
]

const stats = [
  { name: 'Active Analyses', value: '10,000+', description: 'Analyses completed' },
  { name: 'Accuracy Rate', value: '87%', description: 'Prediction accuracy' },
  { name: 'Data Sources', value: '50+', description: 'Financial news sources' },
  { name: 'Response Time', value: '<2s', description: 'Average API response' },
]

export function Landing() {
  return (
    <div className="bg-white dark:bg-gray-900">
      {/* Navigation */}
      <nav className="relative z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 justify-between items-center">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <ChartBarIcon className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    RADGEGRAPH
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Link
                to="/app"
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                Launch App
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-900 dark:to-purple-900/20"></div>
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-20 pb-16 sm:pt-24 sm:pb-20">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-6xl">
                RADGEGRAPH{' '}
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Financial Analysis Engine
                </span>
              </h1>
              <p className="mt-4 text-sm font-semibold uppercase tracking-[0.3em] text-gray-500 dark:text-gray-400">
                Risk and Asset Decision Engine Based on AGENTS, AI, and Knowledge Graph
              </p>
              <p className="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                Make smarter investment decisions with our multi-agent AI system that analyzes market data,
                news sentiment, and financial reports to provide comprehensive trading recommendations.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="mt-10 flex items-center justify-center gap-x-6"
            >
              <Link
                to="/app"
                className="group relative inline-flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 px-8 py-4 text-lg font-semibold text-white shadow-lg hover:from-blue-700 hover:to-purple-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 rounded-xl transition-all duration-200 hover:shadow-xl hover:-translate-y-1"
              >
                <SparklesIcon className="w-5 h-5" />
                Start Analyzing
                <ArrowRightIcon className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <a
                href="#features"
                className="text-sm font-semibold leading-6 text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Learn more <span aria-hidden="true">→</span>
              </a>
            </motion.div>
          </div>

          {/* Hero Image/Dashboard Preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.4 }}
            className="mt-16 flow-root sm:mt-24"
          >
            <div className="relative rounded-xl bg-gray-900/5 p-2 ring-1 ring-inset ring-gray-900/10 lg:rounded-2xl lg:p-4">
              <div className="aspect-video rounded-md bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-pink-500/20 shadow-2xl ring-1 ring-gray-900/10">
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <p className="text-lg font-semibold text-gray-600 dark:text-gray-300">
                      Interactive Dashboard Preview
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                      Real-time financial analysis and recommendations
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-white dark:bg-gray-800 py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:max-w-none">
            <div className="text-center">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-4xl">
                Trusted by thousands of investors
              </h2>
              <p className="mt-4 text-lg leading-8 text-gray-600 dark:text-gray-300">
                Our AI-powered platform delivers consistent results and actionable insights.
              </p>
            </div>
            <dl className="mt-16 grid grid-cols-1 gap-0.5 overflow-hidden rounded-2xl text-center sm:grid-cols-2 lg:grid-cols-4">
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="flex flex-col bg-gray-400/5 p-8"
                >
                  <dt className="text-sm font-semibold leading-6 text-gray-600 dark:text-gray-400">
                    {stat.name}
                  </dt>
                  <dd className="order-first text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
                    {stat.value}
                  </dd>
                  <dd className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {stat.description}
                  </dd>
                </motion.div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="bg-gray-50 dark:bg-gray-900 py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-base font-semibold leading-7 text-blue-600 dark:text-blue-400">
              Advanced Analytics
            </h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-4xl">
              Everything you need for smarter investing
            </p>
            <p className="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-300">
              Our comprehensive platform combines multiple AI agents to deliver unparalleled financial insights.
            </p>
          </div>

          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-2">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="flex flex-col"
                >
                  <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 dark:text-white">
                    <div className={`flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-r ${feature.gradient}`}>
                      <feature.icon className="h-6 w-6 text-white" aria-hidden="true" />
                    </div>
                    {feature.name}
                  </dt>
                  <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600 dark:text-gray-300">
                    <p className="flex-auto">{feature.description}</p>
                  </dd>
                </motion.div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-white dark:bg-gray-800">
        <div className="px-6 py-24 sm:px-6 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-4xl">
              Ready to start investing smarter?
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-gray-600 dark:text-gray-300">
              Join thousands of investors who use RADGEGRAPH to make data-driven investment decisions.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                to="/app"
                className="bg-gradient-to-r from-blue-600 to-purple-600 px-8 py-4 text-lg font-semibold text-white shadow-lg hover:from-blue-700 hover:to-purple-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 rounded-xl transition-all duration-200 hover:shadow-xl hover:-translate-y-1"
              >
                Get started for free
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900">
        <div className="mx-auto max-w-7xl px-6 py-12 md:flex md:items-center md:justify-between lg:px-8">
          <div className="flex justify-center space-x-6 md:order-2">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded flex items-center justify-center">
                <ChartBarIcon className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-medium text-white">RADGEGRAPH</span>
            </div>
          </div>
          <div className="mt-8 md:order-1 md:mt-0">
            <p className="text-center text-xs leading-5 text-gray-400">
              &copy; 2024 RADGEGRAPH Financial Advisor. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
