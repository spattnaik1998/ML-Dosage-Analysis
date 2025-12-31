import { PredictionResponse } from '../types'

interface PredictionCardProps {
  result: PredictionResponse
}

export default function PredictionCard({ result }: PredictionCardProps) {
  const getPredictionColor = (prediction: string) => {
    if (prediction === '<5 days') return 'text-green-700 bg-green-50 border-green-200'
    if (prediction === '5-10 days') return 'text-yellow-700 bg-yellow-50 border-yellow-200'
    return 'text-red-700 bg-red-50 border-red-200'
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-700'
    if (confidence >= 0.6) return 'text-yellow-700'
    return 'text-red-700'
  }

  return (
    <div className="card">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-1">Prediction Result</h2>
          <p className="text-sm text-gray-600">Expected antibiotic treatment duration</p>
        </div>

        {result.requires_human_review && (
          <div className="badge badge-warning flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            Human Review Recommended
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Predicted Duration */}
        <div className={`p-6 rounded-lg border-2 ${getPredictionColor(result.prediction)}`}>
          <div className="text-sm font-medium uppercase tracking-wide mb-2 opacity-75">
            Predicted Duration
          </div>
          <div className="text-4xl font-bold mb-2">
            {result.prediction}
          </div>
          <div className={`text-lg font-semibold ${getConfidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}% confidence
          </div>
        </div>

        {/* Probability Breakdown */}
        <div className="p-6 bg-gray-50 rounded-lg border border-gray-200">
          <div className="text-sm font-medium text-gray-700 mb-4">Class Probabilities</div>
          <div className="space-y-3">
            {Object.entries(result.probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([className, probability]) => (
                <div key={className}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-gray-700">{className}</span>
                    <span className="text-gray-600">{(probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-clinical-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${probability * 100}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
