import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.logging import log

def load_latest_benchmark_csv(results_dir: str = "results/benchmarks") -> pd.DataFrame | None:
    """
    Loads the most recent benchmark summary CSV file.
    """
    if not os.path.exists(results_dir):
        return None
        
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv") and "benchmark_summary" in f]
    if not csv_files:
        return None
        
    csv_files.sort(reverse=True)
    latest_file = os.path.join(results_dir, csv_files[0])
    try:
        df = pd.read_csv(latest_file)
        return df
    except Exception as e:
        log.error(f"Error loading plot data {latest_file}: {e}")
        return None

def plot_task_performance(df: pd.DataFrame, task: str) -> go.Figure | None:
    """
    Creates a bar chart comparing models for a specific task across languages.
    """
    task_df = df[df["Task"] == task]
    if task_df.empty:
        return None
        
    # Determine primary metric based on task
    if task == "summarization":
        metric = "rougeL"
    elif task == "translation":
        metric = "sacrebleu"
    elif task == "qa":
        metric = "f1"
    elif task == "sentiment":
        metric = "macro_f1"
    else:
        metric = task_df.select_dtypes(include='number').columns[0]
        
    fig = px.bar(
        task_df, 
        x='Model', 
        y=metric,
        color='Language',
        barmode='group',
        title=f'{task.title()} Performance ({metric.upper()})',
        labels={metric: f"Score ({metric})"}
    )
    fig.update_layout(xaxis_title="Models", yaxis_title="Score")
    return fig

def plot_language_breakdown(df: pd.DataFrame, language: str) -> go.Figure | None:
    """
    Creates a radar or grouped bar chart comparing models on all tasks for a specific language.
    """
    lang_df = df[df["Language"] == language]
    if lang_df.empty:
        return None
        
    # We must normalize metrics (0-1 range) to plot them together sensibly if they are wildly different.
    # SacreBLEU is typically 0-100, so we normalize
    norm_df = lang_df.copy()
    if 'sacrebleu' in norm_df.columns:
         norm_df['sacrebleu'] = norm_df['sacrebleu'] / 100.0

    # Melt dataframe for easy grouped plotting of multiple varying metrics
    melt_cols = ['Model', 'Task']
    val_cols = [c for c in norm_df.columns if c in ['rougeL', 'f1', 'accuracy', 'sacrebleu']]
    
    if not val_cols:
         return None
         
    melted = pd.melt(norm_df, id_vars=melt_cols, value_vars=val_cols, var_name="Metric", value_name="Score")
    
    # Filter out 0.0 or NaNs that don't belong to the given task
    melted = melted[melted["Score"].notnull()]
    melted = melted[melted["Score"] > 0]
    
    fig = px.bar(
        melted,
        x="Task",
        y="Score",
        color="Model",
        barmode="group",
        title=f"Cross-Task Performance Breakdown for '{language.upper()}'"
    )
    return fig
