import { useState } from 'react'

interface NormalizedFeaturesProps {
  features: Record<string, number>
}

export default function NormalizedFeatures({ features }: NormalizedFeaturesProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const getFeatureDescription = (name: string): string => {
    const descriptions: Record<string, string> = {
      'Age': 'Patient age in years',
      'Dosage (gram)': 'Antibiotic dosage normalized to grams',
      'Gender_Male': 'Binary: 1 = Male, 0 = Female',
      'Route_IV': 'Binary: 1 = Intravenous route',
      'Route_Oral': 'Binary: 1 = Oral route',
      'Frequency_OD': 'Binary: 1 = Once daily (OD)',
      'Frequency_QID': 'Binary: 1 = Four times daily (QID)',
      'Frequency_TDS': 'Binary: 1 = Three times daily (TDS)'
    }
    return descriptions[name] || ''
  }

  const isBinaryFeature = (name: string): boolean => {
    return name.startsWith('Gender_') || name.startsWith('Route_') || name.startsWith('Frequency_')
  }

  const formatValue = (name: string, value: number): string => {
    if (isBinaryFeature(name)) {
      return value === 1 ? 'Yes' : 'No'
    }
    return value.toFixed(2)
  }

  return (
    <div className="card">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between card-header cursor-pointer hover:text-hospital-600 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span>Normalized Model Input</span>
          <span className="badge bg-green-100 text-green-800">
            8 Features
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
        <div className="space-y-3">
          {Object.entries(features).map(([name, value]) => (
            <div key={name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex-1">
                <div className="font-medium text-gray-800">{name}</div>
                <div className="text-xs text-gray-500 mt-0.5">{getFeatureDescription(name)}</div>
              </div>

              <div className="flex items-center gap-3">
                {isBinaryFeature(name) && (
                  <div className={`w-3 h-3 rounded-full ${value === 1 ? 'bg-green-500' : 'bg-gray-300'}`} />
                )}
                <span className={`font-mono font-semibold ${isBinaryFeature(name) ? 'text-gray-600' : 'text-hospital-700'}`}>
                  {formatValue(name, value)}
                </span>
              </div>
            </div>
          ))}

          {Object.keys(features).length === 0 && (
            <p className="text-gray-500 text-center py-4">No normalized features available</p>
          )}

          {/* BD Encoding Note */}
          {features['Route_IV'] === 0 && features['Route_Oral'] === 0 && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex gap-2">
                <svg className="w-5 h-5 text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="text-xs text-blue-800">
                  <strong>BD (Twice Daily) Route:</strong> Both Route flags are 0, indicating BD/other route
                </div>
              </div>
            </div>
          )}

          {features['Frequency_OD'] === 0 && features['Frequency_QID'] === 0 && features['Frequency_TDS'] === 0 && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex gap-2">
                <svg className="w-5 h-5 text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="text-xs text-blue-800">
                  <strong>BD (Twice Daily) Frequency:</strong> All frequency flags are 0, indicating BD frequency
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
