import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from src.visualization.plots import load_latest_benchmark_csv
from src.utils.logging import log

# Set page configuration for a premium dark-mode feel
st.set_page_config(
    page_title="Deep-Indic: Research Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for glassmorphism and modern UI (Injecting style.css if it exists)
if os.path.exists("src/visualization/style.css"):
    with open("src/visualization/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0F172A;
            color: #E2E8F0;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("🔬 Deep-Indic: Research Evaluation Suite")
st.markdown("""
    This dashboard provides research-grade insights into LLM performance on Indian languages, 
    linking **Semantic Similarity** with **Linguistic Complexity**.
""")

# Sidebar for results selection
results_dir = "results/benchmarks"
df = load_latest_benchmark_csv(results_dir)

if df is None or df.empty:
    st.warning("No benchmark results found. Please run the benchmarks first.")
    st.stop()

# Helper to find latest sample-level metrics (Research Depth)
def load_sample_metrics(results_dir: str):
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv") and "sample_level_metrics" in f]
    if not csv_files:
        return None
    csv_files.sort(reverse=True)
    return pd.read_csv(os.path.join(results_dir, csv_files[0]))

sample_df = load_sample_metrics(results_dir)

# Dashboard Layout
tabs = st.tabs(["Live Monitor 🔴", "Overview", "Linguistic Analysis", "Semantic vs. Lexical Gap", "Sample Explorer"])

with tabs[0]:
    st.header("Real-Time Benchmark Monitor")
    progress_file = "results/live_progress.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                p_data = json.load(f)
            
            # Status badge
            status = p_data.get("status", "unknown").upper()
            st.metric("System Status", status)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write(f"**Model:** {p_data.get('current_model', 'N/A')}")
            with col_b:
                st.write(f"**Task:** {p_data.get('current_task', 'N/A')}")
            with col_c:
                st.write(f"**Language:** {p_data.get('current_lang', 'N/A')}")
            
            st.caption(f"Last heartbeat: {p_data.get('last_update', 'N/A')}")
            
            if status == "PROCESSING":
                st.info("📊 Benchmark is actively running. Refresh the page to see latest status.")
            elif status == "COMPLETED":
                st.success("✅ Benchmark execution finished successfully.")
            
        except Exception as e:
            st.error(f"Error reading live feed: {e}")
    else:
        st.info("No active benchmark monitor detected. Run `benchmark_runner.py` to start.")

with tabs[1]:
    st.header("Global Performance Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        task = st.selectbox("Select Task", df["Task"].unique())
        # Filter metrics
        metrics = [c for c in df.columns if c in ['rougeL', 'f1', 'accuracy', 'sacrebleu', 'bert_score', 'chrf++']]
        primary_metric = st.selectbox("Select Metric", metrics)
        
        task_df = df[df["Task"] == task]
        fig = px.bar(
            task_df, x="Model", y=primary_metric, color="Language", barmode="group",
            template="plotly_dark", title=f"{task.title()} across Languages"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        lang = st.selectbox("Select Language", df["Language"].unique())
        lang_df = df[df["Language"] == lang]
        fig_lang = px.bar(
            lang_df, x="Task", y=primary_metric, color="Model", barmode="group",
            template="plotly_dark", title=f"Model performance in {lang.upper()}"
        )
        st.plotly_chart(fig_lang, use_container_width=True)

with tabs[1]:
    st.header("Performance vs. Linguistic Complexity")
    if sample_df is not None:
        st.markdown("Analyzing how models behave as input complexity increases.")
        
        complexity_metric = st.selectbox("Select Complexity Metric", 
                                        ["avg_sentence_length", "avg_token_length", "lexical_diversity_ttr"])
        perf_metric = st.selectbox("Select Performance Metric", 
                                  [c for c in sample_df.columns if c in ['f1', 'exact_match', 'bert_score', 'rougeL']])
        
        fig_scatter = px.scatter(
            sample_df, x=complexity_metric, y=perf_metric, color="Model", 
            marginal_x="box", marginal_y="violin", hover_data=["Reference"],
            trendline="lowess", template="plotly_dark",
            title=f"Correlation: {perf_metric} vs. {complexity_metric}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info("💡 Research Insight: Steeper negative trendlines indicate models that are fragile to linguistic complexity.")
    else:
        st.info("Run the research runner to generate sample-level complexity data.")

with tabs[2]:
    st.header("Semantic Similarity (BERTScore) vs. Lexical Overlap")
    if "bert_score" in df.columns:
        # Comparison of BERTScore vs traditional metrics
        traditional_metric = "rougeL" if "rougeL" in df.columns else "sacrebleu"
        
        fig_gap = px.scatter(
            df, x=traditional_metric, y="bert_score", color="Model", symbol="Task",
            size="Samples", text="Language", template="plotly_dark",
            title=f"The Semantic-Lexical Gap ({traditional_metric} vs. BERTScore)"
        )
        # Add diagonal line (y=x) for reference (ideal matching)
        # Note: Scaling might be needed if traditional metric is 0-100
        st.plotly_chart(fig_gap, use_container_width=True)
        
        st.markdown("""
            **Interpretation:**
            - **Above the diagonal**: Models produce semantically correct outputs that don't match the exact words of the reference (Good for research).
            - **Below the diagonal**: Models might be matching keywords but failing on semantic meaning.
        """)
    else:
        st.warning("BERTScore results not found in the summary.")

with tabs[3]:
    st.header("Granular Sample Explorer")
    if sample_df is not None:
        view_cols = ["Model", "Task", "Language", "Reference", "Prediction", "bert_score"]
        st.dataframe(sample_df[view_cols], use_container_width=True)
    else:
        st.info("Sample数据不可用。")
