import { useState } from 'react'
import { ExtractedFeatureDetail } from '../types'

interface ExtractedFeaturesProps {
  features: Record<string, ExtractedFeatureDetail>
  extractionMethod: string
}

export default function ExtractedFeatures({ features, extractionMethod }: ExtractedFeaturesProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.9) return 'badge badge-success'
    if (confidence >= 0.7) return 'badge badge-info'
    if (confidence >= 0.5) return 'badge badge-warning'
    return 'badge badge-error'
  }

  const getMethodBadge = (method: string) => {
    if (method === 'llm') return 'badge bg-purple-100 text-purple-800'
    if (method === 'fallback') return 'badge bg-orange-100 text-orange-800'
    return 'badge bg-blue-100 text-blue-800'
  }

  const formatFieldName = (field: string) => {
    return field.charAt(0).toUpperCase() + field.slice(1)
  }

  return (
    <div className="card">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between card-header cursor-pointer hover:text-hospital-600 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span>Extracted Features</span>
          <span className={getMethodBadge(extractionMethod)}>
            {extractionMethod.toUpperCase()}
          </span>
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
        <div className="space-y-4">
          {Object.entries(features).map(([fieldName, detail]) => (
            <div key={fieldName} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex items-start justify-between mb-2">
                <span className="font-semibold text-gray-800">{formatFieldName(fieldName)}</span>
                <span className={getConfidenceBadge(detail.confidence)}>
                  {(detail.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex">
                  <span className="text-gray-600 w-24">Value:</span>
                  <span className="font-medium text-gray-900">
                    {detail.value !== null ? String(detail.value) : 'N/A'}
                  </span>
                </div>

                {detail.raw_value && detail.raw_value !== String(detail.value) && (
                  <div className="flex">
                    <span className="text-gray-600 w-24">Raw:</span>
                    <span className="text-gray-700">{detail.raw_value}</span>
                  </div>
                )}

                {detail.unit_conversion && (
                  <div className="flex">
                    <span className="text-gray-600 w-24">Conversion:</span>
                    <span className="text-blue-700 text-xs font-mono">{detail.unit_conversion}</span>
                  </div>
                )}

                {detail.evidence && (
                  <div className="mt-2 pt-2 border-t border-gray-300">
                    <span className="text-gray-600 text-xs block mb-1">Evidence:</span>
                    <span className="text-gray-800 italic bg-white px-2 py-1 rounded border border-gray-200 text-xs block">
                      "{detail.evidence}"
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {Object.keys(features).length === 0 && (
            <p className="text-gray-500 text-center py-4">No features extracted</p>
          )}
        </div>
      )}
    </div>
  )
}
