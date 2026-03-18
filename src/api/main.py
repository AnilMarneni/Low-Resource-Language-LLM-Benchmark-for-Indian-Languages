from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import glob

app = FastAPI(title="Indic LLM Benchmark API")

# Allow CORS for Next.js frontend (default port is usually 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = "results/benchmarks"

def get_latest_benchmark_csv():
    if not os.path.exists(RESULTS_DIR):
        return None
    files = glob.glob(os.path.join(RESULTS_DIR, "benchmark_summary_*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

@app.get("/api/runs")
def get_runs():
    latest_file = get_latest_benchmark_csv()
    if not latest_file:
        return {"runs": []}
    
    try:
        df = pd.read_csv(latest_file)
        # Handle nan values by filling them with 0 so JSON serialization doesn't fail
        df = df.fillna(0)
        
        runs = []
        # We try to create an id that is stable
        for idx, row in df.iterrows():
            model = row.get("Model", "Unknown")
            task = row.get("Task", "Unknown")
            lang = row.get("Language", "Unknown")
            
            # Extract metrics dynamically (anything not standard columns)
            standard_cols = {"Model", "Task", "Language", "Samples"}
            metrics = {k: v for k, v in row.items() if k not in standard_cols}
            
            runs.append({
                "id": f"run-{os.path.basename(latest_file)[:8]}-{idx}",
                "model_id": model,
                "dataset": f"indic-{task}-{lang}",  # pseudo-dataset name
                "task": task,
                "language": lang,
                "metrics": metrics,
                "timestamp": os.path.getctime(latest_file)
            })
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
def get_models():
    # In a real app we might load this from configs/models.yaml
    # For now, derive it from the runs to ensure consistency
    latest_file = get_latest_benchmark_csv()
    if not latest_file:
         return {"models": []}
         
    df = pd.read_csv(latest_file)
    models_list = df["Model"].unique().tolist()
    
    models_out = []
    for m in models_list:
        models_out.append({
            "id": m,
            "name": m.replace("_", " ").title(),
            "provider": "Unknown", # Can be improved by reading configs/models.yaml
            "size": "Unknown",
            "tier": "Unknown"
        })
    return {"models": models_out}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
