import sys
import os
import argparse

# Add root to python path dynamically for module resolving
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.evaluation.benchmark_runner import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Summarization Benchmark")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate (e.g., llama_3_8b_instruct)")
    args = parser.parse_args()
    
    run_benchmark(models_list=args.models, tasks_list=["summarization"])
