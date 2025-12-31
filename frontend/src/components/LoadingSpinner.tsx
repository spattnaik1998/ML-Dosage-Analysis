export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative">
        {/* Outer spinning ring */}
        <div className="w-16 h-16 border-4 border-clinical-200 border-t-clinical-600 rounded-full animate-spin" />

        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <svg className="w-6 h-6 text-clinical-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
        </div>
      </div>

      <div className="text-center">
        <p className="text-lg font-medium text-gray-700 mb-1">Processing Case Study</p>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 bg-clinical-600 rounded-full animate-pulse" />
            Extracting features
          </span>
          <span>→</span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 bg-clinical-600 rounded-full animate-pulse delay-100" />
            Validating
          </span>
          <span>→</span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 bg-clinical-600 rounded-full animate-pulse delay-200" />
            Predicting
          </span>
        </div>
      </div>
    </div>
  )
}
