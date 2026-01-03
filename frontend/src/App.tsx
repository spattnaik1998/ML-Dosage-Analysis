import { useState } from 'react'
import axios from 'axios'
import { PredictionResponse, ErrorResponse } from './types'
import FileUpload from './components/FileUpload'
import PredictionCard from './components/PredictionCard'
import ExtractedFeatures from './components/ExtractedFeatures'
import NormalizedFeatures from './components/NormalizedFeatures'
import ValidationMessages from './components/ValidationMessages'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorDisplay from './components/ErrorDisplay'

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<PredictionResponse[] | null>(null)
  const [error, setError] = useState<ErrorResponse | null>(null)

  const handleFileSubmit = async (file: File) => {
    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post<PredictionResponse[]>('/predict-from-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      setResults(response.data)
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
    <div className="min-h-screen bg-gray-100">
      {/* Professional Hospital Header */}
      <header className="hospital-header sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 medical-gradient rounded-xl flex items-center justify-center shadow-lg">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-display font-bold text-white">Clinical ML Decision Support</h1>
                <p className="text-sm text-gray-300">Antibiotic Treatment Duration Prediction System</p>
              </div>
            </div>

            <div className="hidden md:flex items-center gap-3">
              <div className="px-4 py-2 bg-hospital-600 border border-hospital-500 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-white">System Active</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Information Banner */}
        <div className="mb-8 bg-gradient-to-r from-hospital-50 to-gray-50 border-l-4 border-hospital-600 rounded-lg p-6 shadow-sm">
          <div className="flex gap-4">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-hospital-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-base font-semibold text-medical-900 mb-2">How It Works</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li className="flex items-start gap-2">
                  <span className="text-hospital-600 font-bold">1.</span>
                  Upload a clinical document (PDF, DOCX, or TXT) containing patient case studies
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-hospital-600 font-bold">2.</span>
                  AI extracts key features: patient demographics, antibiotic details, and medical context
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-hospital-600 font-bold">3.</span>
                  System validates, normalizes, and predicts treatment duration for each case
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-hospital-600 font-bold">4.</span>
                  Review predictions with confidence scores and validation alerts
                </li>
              </ul>
              <p className="mt-3 text-xs text-gray-600 italic">
                ⚕️ For clinical decision support only. All predictions should be reviewed by qualified healthcare professionals.
              </p>
            </div>
          </div>
        </div>

        {/* File Upload Section */}
        <div className="mb-8">
          <FileUpload onSubmit={handleFileSubmit} isLoading={isLoading} />
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center py-16">
            <LoadingSpinner />
          </div>
        )}

        {/* Error Display */}
        {error && !isLoading && (
          <div className="mb-8">
            <ErrorDisplay error={error} />
          </div>
        )}

        {/* Results */}
        {results && results.length > 0 && !isLoading && (
          <div className="space-y-8">
            {/* Summary Header */}
            <div className="card bg-gradient-to-r from-hospital-700 to-medical-900 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-display font-bold mb-2">Analysis Complete</h2>
                  <p className="text-gray-200">
                    {results.length} case {results.length === 1 ? 'study' : 'studies'} processed from document
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-4xl font-display font-bold">{results.length}</div>
                  <div className="text-sm text-gray-300">Case Studies</div>
                </div>
              </div>
            </div>

            {/* Individual Case Results */}
            {results.map((result, index) => (
              <div key={index} className="space-y-6">
                {/* Case Header */}
                <div className="flex items-center gap-3 pb-3 border-b-2 border-hospital-200">
                  <div className="w-10 h-10 bg-hospital-600 rounded-lg flex items-center justify-center text-white font-bold">
                    {index + 1}
                  </div>
                  <div>
                    <h3 className="text-xl font-display font-bold text-hospital-900">
                      Case Study #{index + 1}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {result.extraction_method === 'llm_multi_case' ? 'AI-Extracted from Document' : 'Processed'}
                    </p>
                  </div>
                </div>

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

                {/* Case Metadata */}
                <div className="card bg-gray-50">
                  <div className="flex items-center justify-between text-sm text-gray-600">
                    <div className="flex gap-6">
                      <span>
                        <span className="font-medium text-gray-800">Inference Time:</span> {result.inference_time_ms.toFixed(2)}ms
                      </span>
                      <span>
                        <span className="font-medium text-gray-800">Model Version:</span> {result.model_version}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Separator between cases */}
                {index < results.length - 1 && (
                  <div className="my-8 border-t-2 border-dashed border-gray-300"></div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Empty State */}
        {!results && !isLoading && !error && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-hospital-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-10 h-10 text-hospital-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Ready to Process Clinical Documents</h3>
            <p className="text-gray-600 max-w-md mx-auto">
              Upload a document containing patient case studies to begin analysis
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-gray-300 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-6">
            <div>
              <h4 className="font-semibold text-medical-900 mb-3">Pipeline Stages</h4>
              <ul className="text-sm text-gray-600 space-y-2">
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-600 rounded-full"></div>
                  Document Parsing
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-600 rounded-full"></div>
                  LLM Feature Extraction
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-600 rounded-full"></div>
                  Validation & Normalization
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-600 rounded-full"></div>
                  ML Prediction
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-medical-900 mb-3">Supported Formats</h4>
              <ul className="text-sm text-gray-600 space-y-2">
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-700 rounded-full"></div>
                  PDF Documents (.pdf)
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-700 rounded-full"></div>
                  Word Documents (.docx, .doc)
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-700 rounded-full"></div>
                  Text Files (.txt, .md)
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-hospital-700 rounded-full"></div>
                  Multi-case Documents
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-medical-900 mb-3">Model Information</h4>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>Decision Tree Classifier</li>
                <li>8 Clinical Features</li>
                <li>3 Duration Classes</li>
                <li>Production-Grade Pipeline</li>
              </ul>
            </div>
          </div>

          <div className="pt-6 border-t border-gray-200">
            <p className="text-center text-sm text-gray-500">
              Clinical ML Pipeline: Document Upload → Multi-Case Extraction → Validation → Normalization → Inference
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
