# Hunger Detector

A full-stack web application that analyzes CSV files to determine hunger status. Upload a B4 CSV file and get an instant prediction of whether you're hungry or not.

## Features

- 📁 CSV file upload and analysis
- 🎨 Beautiful glassmorphism UI with dark/light theme toggle
- 🌓 Persistent theme preference (stored in localStorage)
- ⚡ Fast and responsive React + Vite frontend
- 🚀 FastAPI backend with CORS support
- 🎯 Simple rule-based prediction (ready to be replaced with ML model)

## Tech Stack

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: FastAPI (Python)
- **Styling**: Tailwind CSS with custom gradients and glassmorphism effects

## Project Structure

```
Aavishkar/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── main.jsx         # React entry point
│   │   └── index.css        # Tailwind CSS imports
│   ├── index.html           # HTML template
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.js       # Vite configuration
│   ├── tailwind.config.js   # Tailwind configuration
│   └── postcss.config.js    # PostCSS configuration
└── README.md                # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Node.js 16+ and npm installed

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

   The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## Usage

1. Make sure both backend and frontend servers are running
2. Open your browser and navigate to `http://localhost:5173`
3. Click "Select CSV File" and choose a CSV file (e.g., `Ayush_B4_08122025.csv`)
4. Click "Analyze" to process the file
5. View the result: "You are hungry" or "You are not hungry"

## How It Works

### Current Prediction Logic

The backend currently uses a simple rule-based approach:
- If the CSV file has **more than 1000 data rows** → "hungry"
- Otherwise → "not_hungry"

### Adding Your ML Model

To replace the simple rule with your ML model:

1. Open `backend/main.py`
2. Locate the `predict_hunger_from_csv()` function
3. Replace the logic with your ML model inference code
4. The function receives `row_count` (number of data rows) and should return `"hungry"` or `"not_hungry"`

Example structure:
```python
def predict_hunger_from_csv(row_count: int) -> str:
    # Load your trained model
    # Extract features from CSV (you may need to pass the CSV data instead of just row_count)
    # Run inference
    # Return "hungry" or "not_hungry"
    pass
```

## API Endpoints

### `GET /`
Returns a simple message confirming the API is running.

**Response:**
```json
{
  "message": "Hunger detection API"
}
```

### `POST /predict`
Accepts a CSV file upload and returns the hunger status prediction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: CSV file (form field: `file`)

**Response:**
```json
{
  "status": "hungry" | "not_hungry"
}
```

**Error Response (400):**
```json
{
  "detail": "Error message"
}
```

## Development Notes

- The frontend uses Tailwind CSS with dark mode support via the `dark` class
- Theme preference is stored in `localStorage` and persists across page reloads
- CORS is configured to allow requests from `http://localhost:5173`
- The UI features glassmorphism effects with backdrop blur
- Error handling is implemented for file validation and API errors

## Future Enhancements

- Add CSV data visualization (charts, statistics)
- Implement proper ML model integration
- Add file validation and preview
- Support for multiple file formats
- Add loading animations and progress indicators
