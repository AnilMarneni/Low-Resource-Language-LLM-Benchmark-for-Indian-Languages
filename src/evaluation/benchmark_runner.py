import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from src.utils.logging import log
from src.utils.config_loader import load_yaml_config

from src.data.dataset_loader import BenchmarkDataLoader
from src.models.model_loader import ModelLoader
from src.models.inference_engine import InferenceEngine
from src.evaluation.metrics import compute_metrics
from src.evaluation.complexity_analyzer import ComplexityAnalyzer
from src.utils.progress_monitor import ProgressMonitor

def run_benchmark(models_list: list[str], tasks_list: list[str], config_dir="configs"):
    log.info(f"Starting Multi-Model Multi-Task Benchmark")
    
    # Load Main Configurations
    try:
        benchmark_config = load_yaml_config(os.path.join(config_dir, "benchmark.yaml"))
        tasks_config = load_yaml_config(os.path.join(config_dir, "tasks.yaml"))
        models_config = load_yaml_config(os.path.join(config_dir, "models.yaml"))
    except Exception as e:
        log.error(f"Failed to load standard configurations: {e}")
        sys.exit(1)

    languages = benchmark_config.get("benchmark", {}).get("languages", ["te", "kn"])
    results_dir = benchmark_config.get("benchmark", {}).get("results_dir", "results/benchmarks")
    predictions_dir = benchmark_config.get("benchmark", {}).get("predictions_dir", "results/predictions")
    os.makedirs(results_dir, exist_ok=True)
    
    data_loader = BenchmarkDataLoader()
    complexity_analyzer = ComplexityAnalyzer(languages)
    monitor = ProgressMonitor()
    monitor.reset()
    
    all_results = []
    all_sample_results = [] # For research-level granular analysis

    # Loop through each model requested by user
    for model_id in models_list:
        model_kwargs = models_config.get("models", {}).get(model_id)
        if not model_kwargs:
            log.error(f"Model ID '{model_id}' not found in configs/models.yaml. Skipping.")
            continue
            
        log.info(f"Initializing Model: {model_id}")
        
        # In a generic multi-model runner, catching memory errors is essential
        try:
            loader = ModelLoader(model_kwargs)
            model, tokenizer = loader.load_model_and_tokenizer()
            batch_size = benchmark_config.get("benchmark", {}).get("batch_size", 4)
            engine = InferenceEngine(model, tokenizer, batch_size=batch_size)
        except Exception as e:
            log.error(f"Failed to load model {model_id} cleanly: {e}")
            continue

        for task in tasks_list:
            if task not in tasks_config.get("tasks", {}):
                log.warning(f"Task '{task}' not found in tasks.yaml. Skipping.")
                continue
                
            for lang in languages:
                monitor.update(
                    current_model=model_id,
                    current_task=task,
                    current_lang=lang,
                    status="processing"
                )
                log.info(f"--- Running: [{model_id}] -> Task: [{task}] -> Lang: [{lang}] ---")
                
                # Fetch formatted dataset
                test_limit = benchmark_config.get("benchmark", {}).get("samples_per_task") 
                # (For quick testing/demo purposes, we might slice this down in arguments but follow global configs)
                dataset = data_loader.get_task_data(task, lang, limit=test_limit)
                
                if dataset is None or len(dataset) == 0:
                    log.warning(f"No valid dataset found for {task} ({lang}). Skipping evaluation.")
                    continue

                # Run Inference to get predictions JSONL
                model_pred_dir = os.path.join(predictions_dir, model_id)
                os.makedirs(model_pred_dir, exist_ok=True)
                
                prediction_file = engine.run_inference(
                    dataset=dataset, 
                    task=task, 
                    config=tasks_config, 
                    output_dir=model_pred_dir
                )
                
                # Evaluate metrics on generated predictions
                with open(prediction_file, "r", encoding="utf-8") as f:
                    preds = []
                    refs = []
                    for line in f:
                        data = json.loads(line)
                        preds.append(data.get("prediction", ""))
                        refs.append(data.get("reference_output", ""))
                        
                log.info(f"Computing metrics and complexity for {len(preds)} samples...")
                
                # Analyze Complexity of the references (ground truth)
                complexity_metrics = complexity_analyzer.analyze_samples(refs, lang)
                
                # Evaluate metrics on generated predictions
                task_metrics = compute_metrics(task, preds, refs, lang=lang)
                
                # Format into master result object
                run_res = {
                    "Model": model_id,
                    "Task": task,
                    "Language": lang,
                    "Samples": len(preds)
                }
                run_res.update(task_metrics)
                run_res.update(complexity_metrics)
                all_results.append(run_res)
                
                # Store sample-level results for depth analysis
                for i, (p, r) in enumerate(zip(preds, refs)):
                    s_comp = complexity_analyzer.get_sample_complexity(r, lang)
                    s_metrics = compute_metrics(task, [p], [r], lang=lang)
                    
                    sample_res = {
                        "Model": model_id,
                        "Task": task,
                        "Language": lang,
                        "Sample_Index": i,
                        "Prediction": p,
                        "Reference": r
                    }
                    sample_res.update(s_metrics)
                    sample_res.update(s_comp)
                    all_sample_results.append(sample_res)
                
                log.info(f"Finished {model_id} - {task} - {lang}: {task_metrics}")

        # Cleanly offload model from GPU memory before proceeding to next model loop
        del model
        del tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate Results to Dataframe immediately upon loop completion
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(results_dir, f"benchmark_summary_{timestamp}.csv")
        df.to_csv(out_csv, index=False)
        log.info(f"Aggregated Benchmark Metrics saved to {out_csv}")
        monitor.complete()
        
        # Save sample-level results for research visualization
        if all_sample_results:
            sample_df = pd.DataFrame(all_sample_results)
            sample_csv = os.path.join(results_dir, f"sample_level_metrics_{timestamp}.csv")
            sample_df.to_csv(sample_csv, index=False)
            log.info(f"Sample-level metrics saved to {sample_csv}")

        print("\n--- Benchmark Final Summary ---")
        print(df.to_string())
    else:
        log.warning("No benchmark results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Central runner for LLM Benchmarking")
    parser.add_argument("--models", nargs="+", required=True, help="List of model keys from models.yaml (e.g., llama_3_8b_instruct mistral_7b_instruct)")
    parser.add_argument("--tasks", nargs="+", required=True, help="List of tasks to evaluate (e.g., summarization qa sentiment)")
    parser.add_argument("--config_dir", type=str, default="configs", help="Directory for YAML configurations")
    
    args = parser.parse_args()
    
    run_benchmark(args.models, args.tasks, args.config_dir)
