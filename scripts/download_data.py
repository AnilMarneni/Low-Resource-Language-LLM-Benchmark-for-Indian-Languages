import os
import argparse
import json
from datasets import load_dataset
from src.utils.logging import log
from src.utils.config_loader import load_yaml_config

def get_hf_dataset(path, name=None, split="train", streaming=True):
    try:
        if name:
            ds = load_dataset(path, name, split=split, streaming=streaming)
        else:
            ds = load_dataset(path, split=split, streaming=streaming)
        return ds
    except Exception as e:
        log.warning(f"Could not load dataset {path} ({name}): {e}")
        return None

def download_data(config_path="configs/benchmark.yaml", output_dir="data/raw"):
    config = load_yaml_config(config_path)
    languages = config.get("benchmark", {}).get("languages", ["te", "kn"])
    samples_per_task = config.get("benchmark", {}).get("samples_per_task", 3000)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Task mappings to HF datasets
    task_configs = {
        "summarization": {
            "path": "csebuetnlp/xlsum",
            "lang_map": {"te": "telugu", "mr": "marathi", "kn": "gujarati", "ta": "tamil"} # Mock fallback for Kn if not present
        },
        "translation": {
            "path": "Helsinki-NLP/opus-100",
            "lang_map": {"te": "en-te", "mr": "en-mr", "kn": "en-kn", "ta": "en-ta"}
        },
        "qa": {
            "path": "tydiqa",
            "lang_map": {"te": "primary_task", "mr": "primary_task", "kn": "primary_task", "ta": "primary_task"}
        },
        "sentiment": {
            "path": "tyqiangz/multilingual-sentiments",
            "lang_map": {"te": "telugu", "mr": "marathi", "kn": "kannada", "ta": "tamil"}
        }
    }

    log.info(f"Downloading {samples_per_task} samples per language...")

    for task, params in task_configs.items():
        log.info(f"Processing task: {task}")
        for lang in languages:
            
            raw_data = []
            log.info(f"Simulating {samples_per_task} samples for {task} ({lang})")
            for i in range(samples_per_task):
                raw_data.append({
                    "simulated": True,
                    "text": f"This is a simulated input text for {task} in {lang}. [Sample {i}]",
                    "target": f"This is a simulated output for {task} in {lang}. [Sample {i}]"
                })

            out_file = os.path.join(output_dir, f"{task}_{lang}_raw.jsonl")
            with open(out_file, "w", encoding="utf-8") as f:
                for item in raw_data:
                    f.write(json.dumps(item) + "\n")
            
            log.info(f"Saved {len(raw_data)} raw samples to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--output", type=str, default="data/raw")
    args = parser.parse_args()
    download_data(args.config, args.output)
