import { useState } from 'react'
import axios from 'axios'
import { PredictionResponse, PredictionRequest, ErrorResponse } from './types'
import TextInput from './components/TextInput'
import PredictionCard from './components/PredictionCard'
import ExtractedFeatures from './components/ExtractedFeatures'
import NormalizedFeatures from './components/NormalizedFeatures'
import ValidationMessages from './components/ValidationMessages'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorDisplay from './components/ErrorDisplay'

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<ErrorResponse | null>(null)

  const handleSubmit = async (caseStudyText: string) => {
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post<PredictionResponse>('/predict', {
        case_study_text: caseStudyText
      } as PredictionRequest)

      setResult(response.data)
    } catch (err: any) {
      if (err.response?.data) {
        setError(err.response.data)
      } else {
        setError({
          error: 'Network Error',
          detail: 'Unable to connect to the backend server. Please ensure it is running on port 8000.',
          stage: 'network'
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-clinical-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-clinical-600 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Clinical ML Predictor</h1>
              <p className="text-sm text-gray-600">Antibiotic Treatment Duration Prediction</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Instructions */}
        <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex gap-3">
            <svg className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h3 className="text-sm font-semibold text-blue-900 mb-1">How to use</h3>
              <p className="text-sm text-blue-800">
                Enter clinical case study text describing patient demographics, antibiotic details (drug, dosage, route, frequency),
                and relevant medical context. The system will extract key features, validate them, and predict treatment duration.
              </p>
            </div>
          </div>
        </div>

        {/* Input Section */}
        <div className="mb-6">
          <TextInput onSubmit={handleSubmit} isLoading={isLoading} />
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center py-12">
            <LoadingSpinner />
          </div>
        )}

        {/* Error Display */}
        {error && !isLoading && (
          <div className="mb-6">
            <ErrorDisplay error={error} />
          </div>
        )}

        {/* Results */}
        {result && !isLoading && (
          <div className="space-y-6">
            {/* Main Prediction */}
            <PredictionCard result={result} />

            {/* Pipeline Transparency */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Extracted Features */}
              <ExtractedFeatures
                features={result.extracted_features}
                extractionMethod={result.extraction_method}
              />

              {/* Normalized Features */}
              <NormalizedFeatures features={result.normalized_features} />
            </div>

            {/* Validation Messages */}
            {result.validation_messages.length > 0 && (
              <ValidationMessages
                messages={result.validation_messages}
                requiresHumanReview={result.requires_human_review}
              />
            )}

            {/* Metadata */}
            <div className="card">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <div className="flex gap-6">
                  <span>
                    <span className="font-medium">Inference Time:</span> {result.inference_time_ms.toFixed(2)}ms
                  </span>
                  <span>
                    <span className="font-medium">Model Version:</span> {result.model_version}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!result && !isLoading && !error && (
          <div className="text-center py-12">
            <svg className="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-gray-500 text-lg">Enter a clinical case study to begin</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            ML Pipeline: LLM Extraction → Validation → Normalization → Decision Tree Inference
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
