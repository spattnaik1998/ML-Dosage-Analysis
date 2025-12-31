export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center gap-6">
      <div className="relative">
        {/* Outer spinning ring */}
        <div className="w-20 h-20 border-4 border-hospital-200 border-t-hospital-600 rounded-full animate-spin" />

        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <svg className="w-8 h-8 text-hospital-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
        </div>
      </div>

      <div className="text-center">
        <p className="text-xl font-display font-semibold text-hospital-900 mb-3">Processing Clinical Document</p>
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-center gap-3 text-sm text-gray-600">
            <span className="flex items-center gap-2 px-3 py-1.5 bg-hospital-50 rounded-lg">
              <span className="inline-block w-2 h-2 bg-hospital-600 rounded-full animate-pulse" />
              Parsing document
            </span>
            <span className="text-gray-400">→</span>
            <span className="flex items-center gap-2 px-3 py-1.5 bg-hospital-50 rounded-lg">
              <span className="inline-block w-2 h-2 bg-hospital-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}} />
              Extracting features
            </span>
          </div>
          <div className="flex items-center justify-center gap-3 text-sm text-gray-600">
            <span className="flex items-center gap-2 px-3 py-1.5 bg-hospital-50 rounded-lg">
              <span className="inline-block w-2 h-2 bg-hospital-600 rounded-full animate-pulse" style={{animationDelay: '0.4s'}} />
              Validating data
            </span>
            <span className="text-gray-400">→</span>
            <span className="flex items-center gap-2 px-3 py-1.5 bg-hospital-50 rounded-lg">
              <span className="inline-block w-2 h-2 bg-hospital-600 rounded-full animate-pulse" style={{animationDelay: '0.6s'}} />
              Running predictions
            </span>
          </div>
        </div>
        <p className="mt-4 text-xs text-gray-500 italic">This may take a few moments for multi-case documents</p>
      </div>
    </div>
  )
}
