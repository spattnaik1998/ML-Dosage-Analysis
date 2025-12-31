import { useState } from 'react'

interface TextInputProps {
  onSubmit: (text: string) => void
  isLoading: boolean
}

const EXAMPLE_TEXT = `Patient is a 65-year-old male admitted with severe community-acquired pneumonia. Medical history includes type 2 diabetes and hypertension. Started on intravenous ceftriaxone 2 grams once daily. Patient is responding well to treatment with improving respiratory symptoms and decreasing inflammatory markers.`

export default function TextInput({ onSubmit, isLoading }: TextInputProps) {
  const [text, setText] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (text.trim()) {
      onSubmit(text.trim())
    }
  }

  const loadExample = () => {
    setText(EXAMPLE_TEXT)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Clinical Case Study</h2>
        <button
          type="button"
          onClick={loadExample}
          className="text-sm text-clinical-600 hover:text-clinical-700 font-medium"
        >
          Load Example
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter clinical case study text here. Include patient age, gender, antibiotic details (name, dosage, route, frequency), and relevant medical context..."
          className="w-full h-48 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-clinical-500 focus:border-transparent resize-none"
          disabled={isLoading}
        />

        <div className="flex items-center justify-between mt-4">
          <span className="text-sm text-gray-500">
            {text.length} / 5000 characters
          </span>

          <button
            type="submit"
            disabled={isLoading || !text.trim()}
            className="btn-primary"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Processing...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Predict Duration
              </span>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
