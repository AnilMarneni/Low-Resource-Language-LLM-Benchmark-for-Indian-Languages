import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.evaluation.benchmark_runner import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis Benchmark")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate")
    args = parser.parse_args()
    
    run_benchmark(models_list=args.models, tasks_list=["sentiment"])
