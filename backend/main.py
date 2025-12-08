from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import csv
import io
from typing import Dict

app = FastAPI(title="Hunger Detection API")

# Enable CORS for frontend (Vite default port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_hunger_from_csv(row_count: int) -> str:
    """
    Simple rule-based prediction for hunger status.
    
    TODO: Replace this with your ML model logic.
    For example:
    - Load a trained model
    - Extract features from CSV rows
    - Run inference
    - Return prediction
    
    Args:
        row_count: Number of data rows in the CSV (excluding header)
    
    Returns:
        "hungry" or "not_hungry"
    """
    if row_count > 1000:
        return "hungry"
    else:
        return "not_hungry"


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hunger detection API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Predict hunger status from uploaded CSV file.
    
    Args:
        file: CSV file upload
    
    Returns:
        JSON with status: "hungry" or "not_hungry"
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read file content
        contents = await file.read()
        contents_str = contents.decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.reader(io.StringIO(contents_str))
        
        # Count data rows (excluding header)
        rows = list(csv_reader)
        if len(rows) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # First row is header, so data rows = total - 1
        data_row_count = len(rows) - 1
        
        # Get prediction
        status = predict_hunger_from_csv(data_row_count)
        
        return {"status": status}
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding error. Please ensure CSV is UTF-8 encoded.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

