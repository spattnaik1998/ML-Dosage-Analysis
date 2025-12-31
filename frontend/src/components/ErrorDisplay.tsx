import { ErrorResponse } from '../types'

interface ErrorDisplayProps {
  error: ErrorResponse
}

export default function ErrorDisplay({ error }: ErrorDisplayProps) {
  const getErrorIcon = (stage: string) => {
    if (stage === 'network') {
      return (
        <svg className="w-12 h-12 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414" />
        </svg>
      )
    }
    return (
      <svg className="w-12 h-12 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  }

  const getStageDescription = (stage: string) => {
    const stages: Record<string, string> = {
      'extraction': 'Feature extraction from text failed',
      'normalization': 'Feature normalization failed',
      'inference': 'Model prediction failed',
      'network': 'Unable to connect to backend server',
      'unknown': 'An unexpected error occurred'
    }
    return stages[stage] || stages['unknown']
  }

  return (
    <div className="card border-2 border-red-200 bg-red-50">
      <div className="flex gap-4">
        <div className="flex-shrink-0">
          {getErrorIcon(error.stage)}
        </div>

        <div className="flex-1">
          <h3 className="text-lg font-bold text-red-900 mb-2">
            {error.error}
          </h3>

          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <span className="badge badge-error">{error.stage.toUpperCase()}</span>
              <p className="text-sm text-red-800 flex-1">
                {getStageDescription(error.stage)}
              </p>
            </div>

            {error.detail && (
              <div className="mt-3 p-3 bg-white rounded border border-red-200">
                <p className="text-sm text-gray-700 font-mono">{error.detail}</p>
              </div>
            )}
          </div>

          {error.stage === 'network' && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-sm text-blue-900">
                <strong>Troubleshooting:</strong>
              </p>
              <ul className="text-sm text-blue-800 mt-2 space-y-1 list-disc list-inside">
                <li>Ensure the backend server is running on port 8000</li>
                <li>Check that API keys are configured in .env file</li>
                <li>Verify network connectivity</li>
              </ul>
            </div>
          )}

          {error.stage === 'extraction' && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-sm text-blue-900">
                <strong>Suggestions:</strong>
              </p>
              <ul className="text-sm text-blue-800 mt-2 space-y-1 list-disc list-inside">
                <li>Ensure the text includes patient age, gender, and antibiotic details</li>
                <li>Check that API keys (OPENAI_API_KEY or GOOGLE_API_KEY) are valid</li>
                <li>Try providing more detailed clinical information</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
