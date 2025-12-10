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
      {/* PixelBlast Background - More Visible, Starting from Top */}
      <div className={`fixed top-0 left-0 w-screen h-screen z-0 transition-opacity duration-500 ${
        theme === 'dark' ? 'opacity-70' : 'opacity-60'
      }`}
      style={{
        margin: 0,
        padding: 0
      }}>
        <PixelBlast
          variant="circle"
          pixelSize={5}
          color={theme === 'dark' ? '#C084FC' : '#818CF8'}
          patternScale={2.5}
          patternDensity={1.5}
          pixelSizeJitter={0.6}
          enableRipples
          rippleSpeed={0.5}
          rippleThickness={0.15}
          rippleIntensityScale={2}
          liquid
          liquidStrength={0.15}
          liquidRadius={1.3}
          liquidWobbleSpeed={6}
          speed={0.7}
          edgeFade={0}
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
          {/* Enhanced Glassmorphism Card - True Glass Effect */}
          <div className={`backdrop-blur-3xl rounded-3xl shadow-2xl p-8 border-2 transition-all duration-500 ${
            theme === 'dark'
              ? 'bg-white/5 border-white/20 shadow-purple-500/20'
              : 'bg-white/40 border-white/40 shadow-indigo-500/20'
          }`}
          style={{
            background: theme === 'dark'
              ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%)'
              : 'linear-gradient(135deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0.3) 100%)',
            boxShadow: theme === 'dark' 
              ? '0 8px 32px 0 rgba(192, 132, 252, 0.2), inset 0 1px 0 0 rgba(255, 255, 255, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.1)' 
              : '0 8px 32px 0 rgba(99, 102, 241, 0.2), inset 0 1px 0 0 rgba(255, 255, 255, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.3)',
            backdropFilter: 'blur(40px) saturate(180%)',
            WebkitBackdropFilter: 'blur(40px) saturate(180%)'
          }}>
            {/* Title */}
            <h1 className={`text-5xl font-bold text-center mb-3 bg-gradient-to-r bg-clip-text text-transparent transition-all duration-500 leading-tight pb-2 ${
              theme === 'dark' 
                ? 'from-purple-200 via-pink-200 to-purple-200' 
                : 'from-indigo-600 via-purple-600 to-indigo-600'
            }`}
            style={{
              lineHeight: '1.2',
              paddingBottom: '0.5rem'
            }}>
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
                  file:bg-indigo-500
                  file:text-white file:shadow-lg
                  hover:file:bg-indigo-600
                  file:cursor-pointer file:transition-all file:duration-300
                  file:transform file:hover:scale-105
                  cursor-pointer
                  rounded-xl border-2 p-3 transition-all duration-300 backdrop-blur-sm ${
                    theme === 'dark'
                      ? 'text-slate-300 bg-white/10 border-white/30 hover:border-white/40'
                      : 'text-slate-700 bg-white/60 border-white/50 hover:border-indigo-400'
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
              <div className={`mt-6 p-4 rounded-xl border-2 backdrop-blur-md transition-all duration-300 ${
                theme === 'dark'
                  ? 'bg-red-500/30 border-red-500/60'
                  : 'bg-red-100/90 border-red-400/60'
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
              <div className={`mt-6 p-6 rounded-2xl border-2 backdrop-blur-md transition-all duration-500 ${
                result.status === 'hungry'
                  ? theme === 'dark'
                    ? 'bg-amber-500/30 border-amber-500/60 shadow-amber-500/30'
                    : 'bg-amber-100/90 border-amber-400/60 shadow-amber-500/20'
                  : theme === 'dark'
                    ? 'bg-emerald-500/30 border-emerald-500/60 shadow-emerald-500/30'
                    : 'bg-emerald-100/90 border-emerald-400/60 shadow-emerald-500/20'
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
