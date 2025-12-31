import { useState } from 'react'
import { ValidationMessage } from '../types'

interface ValidationMessagesProps {
  messages: ValidationMessage[]
  requiresHumanReview: boolean
}

export default function ValidationMessages({ messages, requiresHumanReview }: ValidationMessagesProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
      case 'ERROR':
        return {
          badge: 'badge badge-error',
          bg: 'bg-red-50 border-red-200',
          icon: (
            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        }
      case 'WARNING':
        return {
          badge: 'badge badge-warning',
          bg: 'bg-yellow-50 border-yellow-200',
          icon: (
            <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )
        }
      default:
        return {
          badge: 'badge badge-info',
          bg: 'bg-blue-50 border-blue-200',
          icon: (
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        }
    }
  }

  const getCardHeaderColor = () => {
    const hasCritical = messages.some(m => m.severity === 'CRITICAL' || m.severity === 'ERROR')
    const hasWarning = messages.some(m => m.severity === 'WARNING')

    if (hasCritical) return 'text-red-700'
    if (hasWarning) return 'text-yellow-700'
    return 'text-blue-700'
  }

  return (
    <div className="card">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full flex items-center justify-between card-header cursor-pointer hover:opacity-80 transition-opacity ${getCardHeaderColor()}`}
      >
        <div className="flex items-center gap-3">
          <span>Validation & Quality Checks</span>
          <span className="badge bg-gray-100 text-gray-800">
            {messages.length} {messages.length === 1 ? 'message' : 'messages'}
          </span>
          {requiresHumanReview && (
            <span className="badge badge-warning">Review Required</span>
          )}
        </div>
        <svg
          className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isExpanded && (
        <div className="space-y-3">
          {messages.map((message, index) => {
            const styles = getSeverityStyles(message.severity)
            return (
              <div key={index} className={`p-4 rounded-lg border ${styles.bg}`}>
                <div className="flex gap-3">
                  {styles.icon}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={styles.badge}>{message.severity}</span>
                      <span className="text-sm font-semibold text-gray-800">{message.field}</span>
                      {message.repaired && (
                        <span className="badge bg-green-100 text-green-800 text-xs">Auto-repaired</span>
                      )}
                    </div>
                    <p className="text-sm text-gray-700">{message.message}</p>
                  </div>
                </div>
              </div>
            )
          })}

          {requiresHumanReview && (
            <div className="mt-4 p-4 bg-yellow-50 border-2 border-yellow-300 rounded-lg">
              <div className="flex gap-3">
                <svg className="w-6 h-6 text-yellow-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <h4 className="font-semibold text-yellow-900 mb-1">Human Review Recommended</h4>
                  <p className="text-sm text-yellow-800">
                    This prediction contains quality issues that may affect accuracy.
                    Please review the extracted features and validation messages before making clinical decisions.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
