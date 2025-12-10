import { useState, useEffect } from 'react'
import PixelBlast from './components/PixelBlast'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [theme, setTheme] = useState('dark')

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'dark'
    setTheme(savedTheme)
    document.documentElement.classList.toggle('dark', savedTheme === 'dark')
  }, [])

  // Update theme and localStorage
  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.classList.toggle('dark', newTheme === 'dark')
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setResult(null)
      setError(null)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a CSV file first')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to analyze file')
      }

      const data = await response.json()
      
      // Map status to human-readable text
      const statusText = data.status === 'hungry' 
        ? 'You are hungry' 
        : 'You are not hungry'
      
      setResult({
        status: data.status,
        text: statusText,
      })
    } catch (err) {
      setError(err.message || 'An error occurred while analyzing the file')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={`min-h-screen relative overflow-hidden transition-colors duration-500 ${
      theme === 'dark' 
        ? 'bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950' 
        : 'bg-gradient-to-br from-slate-50 via-slate-100 to-slate-50'
    }`}>
      {/* PixelBlast Background */}
      <div className="absolute inset-0 w-full h-full z-0 opacity-30 dark:opacity-20">
        <PixelBlast
          variant="circle"
          pixelSize={6}
          color={theme === 'dark' ? '#B19EEF' : '#6366F1'}
          patternScale={3}
          patternDensity={1.2}
          pixelSizeJitter={0.5}
          enableRipples
          rippleSpeed={0.4}
          rippleThickness={0.12}
          rippleIntensityScale={1.5}
          liquid
          liquidStrength={0.12}
          liquidRadius={1.2}
          liquidWobbleSpeed={5}
          speed={0.6}
          edgeFade={0.25}
          transparent
        />
      </div>

      {/* Theme Toggle */}
      <div className="absolute top-6 right-6 z-50">
        <button
          onClick={toggleTheme}
          className={`p-3 rounded-full backdrop-blur-xl hover:opacity-80 transition-all duration-300 shadow-2xl border ${
            theme === 'dark' 
              ? 'bg-white/10 hover:bg-white/20 border-white/20' 
              : 'bg-white/60 hover:bg-white/80 border-white/40'
          }`}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? (
            <span className="text-2xl transition-transform duration-300 hover:rotate-180">☀️</span>
          ) : (
            <span className="text-2xl transition-transform duration-300 hover:rotate-180">🌙</span>
          )}
        </button>
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex items-center justify-center min-h-screen p-6">
        <div className="w-full max-w-md">
          {/* Enhanced Glassmorphism Card */}
          <div className={`backdrop-blur-2xl rounded-3xl shadow-2xl p-8 border transition-all duration-500 ${
            theme === 'dark'
              ? 'bg-white/5 border-white/10 shadow-purple-500/10'
              : 'bg-white/70 border-white/30 shadow-indigo-500/10'
          }`}
          style={{
            boxShadow: theme === 'dark' 
              ? '0 8px 32px 0 rgba(177, 158, 239, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.1)' 
              : '0 8px 32px 0 rgba(99, 102, 241, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.3)'
          }}>
            {/* Title */}
            <h1 className={`text-5xl font-bold text-center mb-3 bg-gradient-to-r bg-clip-text text-transparent transition-all duration-500 ${
              theme === 'dark' 
                ? 'from-purple-200 via-pink-200 to-purple-200' 
                : 'from-indigo-600 via-purple-600 to-indigo-600'
            }`}>
              Hunger Detector
            </h1>
            <p className={`text-center mb-8 text-lg ${
              theme === 'dark' ? 'text-slate-300' : 'text-slate-600'
            }`}>
              Upload your B4 CSV file to see if you're hungry.
            </p>

            {/* File Input */}
            <div className="mb-6">
              <label className={`block mb-3 text-sm font-semibold ${
                theme === 'dark' ? 'text-slate-200' : 'text-slate-700'
              }`}>
                Select CSV File
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className={`block w-full text-sm
                  file:mr-4 file:py-3 file:px-5
                  file:rounded-xl file:border-0
                  file:text-sm file:font-semibold
                  file:bg-gradient-to-r file:from-indigo-500 file:to-purple-500
                  file:text-white file:shadow-lg
                  hover:file:from-indigo-600 hover:file:to-purple-600
                  file:cursor-pointer file:transition-all file:duration-300
                  file:transform file:hover:scale-105
                  cursor-pointer
                  rounded-xl border-2 p-3 transition-all duration-300 ${
                    theme === 'dark'
                      ? 'text-slate-300 bg-white/5 border-white/20 hover:border-white/30'
                      : 'text-slate-700 bg-white/50 border-slate-300 hover:border-indigo-400'
                  }`}
              />
              {selectedFile && (
                <p className={`mt-3 text-sm font-medium ${
                  theme === 'dark' ? 'text-slate-400' : 'text-slate-600'
                }`}>
                  ✓ Selected: {selectedFile.name}
                </p>
              )}
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={isLoading || !selectedFile}
              className={`w-full py-4 px-6 rounded-xl font-bold text-lg
                disabled:cursor-not-allowed
                transition-all duration-300 shadow-xl
                transform hover:scale-[1.02] active:scale-[0.98]
                ${
                  isLoading || !selectedFile
                    ? 'bg-slate-600 text-slate-400'
                    : 'bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 hover:from-indigo-700 hover:via-purple-700 hover:to-indigo-700 text-white shadow-indigo-500/50'
                }`}
            >
              {isLoading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </span>
              ) : (
                'Analyze'
              )}
            </button>

            {/* Error Display */}
            {error && (
              <div className={`mt-6 p-4 rounded-xl border-2 backdrop-blur-sm transition-all duration-300 ${
                theme === 'dark'
                  ? 'bg-red-500/20 border-red-500/50'
                  : 'bg-red-100/80 border-red-400/50'
              }`}>
                <p className={`text-sm font-medium ${
                  theme === 'dark' ? 'text-red-200' : 'text-red-800'
                }`}>
                  {error}
                </p>
              </div>
            )}

            {/* Result Display */}
            {result && (
              <div className={`mt-6 p-6 rounded-2xl border-2 backdrop-blur-sm transition-all duration-500 ${
                result.status === 'hungry'
                  ? theme === 'dark'
                    ? 'bg-amber-500/20 border-amber-500/50 shadow-amber-500/20'
                    : 'bg-amber-100/80 border-amber-400/50 shadow-amber-500/10'
                  : theme === 'dark'
                    ? 'bg-emerald-500/20 border-emerald-500/50 shadow-emerald-500/20'
                    : 'bg-emerald-100/80 border-emerald-400/50 shadow-emerald-500/10'
              }`}
              style={{
                animation: 'fadeIn 0.5s ease-in'
              }}>
                <p className={`text-center text-2xl font-bold ${
                  result.status === 'hungry'
                    ? theme === 'dark' ? 'text-amber-200' : 'text-amber-800'
                    : theme === 'dark' ? 'text-emerald-200' : 'text-emerald-800'
                }`}>
                  {result.text}
                </p>
              </div>
            )}

            {/* TODO: Add CSV parsing visualization here if needed */}
            {/* TODO: Add charts/stats display for CSV data */}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  )
}

export default App
