import { useState, useEffect } from 'react'

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
    <div className={`min-h-screen bg-gradient-to-br transition-colors duration-300 ${
      theme === 'dark' 
        ? 'from-slate-950 via-slate-900 to-slate-950' 
        : 'from-slate-50 via-slate-100 to-slate-50'
    }`}>
      {/* Theme Toggle */}
      <div className="absolute top-6 right-6">
        <button
          onClick={toggleTheme}
          className={`p-3 rounded-full backdrop-blur-md hover:opacity-80 transition-all duration-200 shadow-lg ${
            theme === 'dark' 
              ? 'bg-white/10 hover:bg-white/20' 
              : 'bg-slate-200/80 hover:bg-slate-300/80'
          }`}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? (
            <span className="text-2xl">☀️</span>
          ) : (
            <span className="text-2xl">🌙</span>
          )}
        </button>
      </div>

      {/* Main Content */}
      <div className="flex items-center justify-center min-h-screen p-6">
        <div className="w-full max-w-md">
          {/* Glassmorphism Card */}
          <div className={`backdrop-blur-xl rounded-2xl shadow-xl p-8 border transition-colors duration-300 ${
            theme === 'dark'
              ? 'bg-white/10 border-white/20'
              : 'bg-white/90 border-slate-200/50'
          }`}>
            {/* Title */}
            <h1 className={`text-4xl font-bold text-center mb-2 ${
              theme === 'dark' ? 'text-slate-100' : 'text-slate-900'
            }`}>
              Hunger Detector
            </h1>
            <p className={`text-center mb-8 ${
              theme === 'dark' ? 'text-slate-300' : 'text-slate-600'
            }`}>
              Upload your B4 CSV file to see if you're hungry.
            </p>

            {/* File Input */}
            <div className="mb-6">
              <label className={`block mb-2 text-sm font-medium ${
                theme === 'dark' ? 'text-slate-200' : 'text-slate-700'
              }`}>
                Select CSV File
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className={`block w-full text-sm
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-500 file:text-white
                  hover:file:bg-indigo-600
                  file:cursor-pointer
                  cursor-pointer
                  rounded-lg border p-2 transition-colors duration-300 ${
                    theme === 'dark'
                      ? 'text-slate-300 bg-white/5 border-white/20'
                      : 'text-slate-700 bg-slate-50 border-slate-300'
                  }`}
              />
              {selectedFile && (
                <p className={`mt-2 text-sm ${
                  theme === 'dark' ? 'text-slate-400' : 'text-slate-600'
                }`}>
                  Selected: {selectedFile.name}
                </p>
              )}
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={isLoading || !selectedFile}
              className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-700 
                disabled:bg-slate-600 disabled:cursor-not-allowed
                text-white font-semibold rounded-lg
                transition-all duration-200 shadow-lg
                transform hover:scale-[1.02] active:scale-[0.98]"
            >
              {isLoading ? 'Analyzing...' : 'Analyze'}
            </button>

            {/* Error Display */}
            {error && (
              <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <p className={`text-sm ${
                  theme === 'dark' ? 'text-red-200' : 'text-red-800'
                }`}>
                  {error}
                </p>
              </div>
            )}

            {/* Result Display */}
            {result && (
              <div className={`mt-6 p-6 rounded-lg border-2 ${
                result.status === 'hungry'
                  ? 'bg-amber-500/20 border-amber-500/50'
                  : 'bg-emerald-500/20 border-emerald-500/50'
              }`}>
                <p className={`text-center text-xl font-bold ${
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
    </div>
  )
}

export default App

