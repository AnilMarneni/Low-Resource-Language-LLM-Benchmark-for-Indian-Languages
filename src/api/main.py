from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import glob
import json
from src.utils.logging import log

app = FastAPI(title="Indic LLM Benchmark API")

# Allow CORS for Next.js frontend (default port is usually 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = "results/benchmarks"

def get_latest_file(pattern):
    if not os.path.exists(RESULTS_DIR):
        return None
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

@app.get("/api/runs")
def get_runs():
    latest_file = get_latest_file("benchmark_summary_*.csv")
    if not latest_file:
        return {"runs": []}
    
    try:
        df = pd.read_csv(latest_file).fillna(0)
        runs = []
        for idx, row in df.iterrows():
            standard_cols = {"Model", "Task", "Language", "Samples"}
            metrics = {k: v for k, v in row.items() if k not in standard_cols}
            
            runs.append({
                "id": f"run-{idx}",
                "model_id": row.get("Model", "Unknown"),
                "task": row.get("Task", "Unknown"),
                "language": row.get("Language", "Unknown"),
                "samples": int(row.get("Samples", 0)),
                "metrics": metrics,
                "timestamp": os.path.getctime(latest_file)
            })
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, e):
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred", "details": str(e), "status": "error"}
    )

@app.get("/api/progress")
def get_progress():
    progress_file = "results/live_progress.json"
    if not os.path.exists(progress_file):
        return {"status": "idle", "message": "Ready to run benchmarks."}
    try:
        with open(progress_file, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/research/complexity")
def get_complexity_analysis():
    latest_file = get_latest_file("sample_level_metrics_*.csv")
    if not latest_file:
        return {"samples": []}
    
    try:
        df = pd.read_csv(latest_file).fillna(0)
        # Limit the number of samples sent to the frontend to prevent bloat
        samples = df.head(500).to_dict(orient="records")
        return {"samples": samples}
    except Exception as e:
        log.error(f"Error reading complexity CSV: {e}")
        return {"samples": []}

@app.get("/api/models")
def get_models():
    latest_file = get_latest_file("benchmark_summary_*.csv")
    if not latest_file:
         return {"models": []}
         
    df = pd.read_csv(latest_file)
    models_list = df["Model"].unique().tolist()
    
    models_out = []
    for m in models_list:
        models_out.append({
            "id": m,
            "name": m.replace("_", " ").title(),
            "provider": "Open Source",
            "tier": "Research"
        })
    return {"models": models_out}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
