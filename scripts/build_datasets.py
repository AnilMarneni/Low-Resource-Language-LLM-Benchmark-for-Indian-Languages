import os
import argparse
import json
from src.utils.logging import log
from src.utils.config_loader import load_yaml_config
from src.data.data_preprocessing import DataPreprocessor

def build_datasets(config_path="configs/benchmark.yaml", raw_dir="data/raw", output_dir="data/processed"):
    config = load_yaml_config(config_path)
    languages = config.get("benchmark", {}).get("languages", ["te", "kn"])
    tasks = ["summarization", "translation", "qa", "sentiment"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for task in tasks:
        for lang in languages:
            raw_file = os.path.join(raw_dir, f"{task}_{lang}_raw.jsonl")
            if not os.path.exists(raw_file):
                log.warning(f"Raw file not found: {raw_file}")
                continue
            
            out_file = os.path.join(output_dir, f"{task}_{lang}.jsonl")
            preprocessor = DataPreprocessor(language=lang)
            
            processed_data = []
            seen_inputs = set()
            
            with open(raw_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # Interpret schema based on tasks
                        input_text = ""
                        ref_output = ""
                        
                        if item.get("simulated"):
                            input_text = item.get("text")
                            ref_output = item.get("target")
                        else:
                            if task == "summarization":
                                input_text = item.get("text", "")
                                ref_output = item.get("summary", "")
                            elif task == "translation":
                                trans = item.get("translation", {})
                                input_text = trans.get("en", "")
                                ref_output = trans.get(lang, "")
                            elif task == "qa":
                                input_text = item.get("context", "") + " " + item.get("question", "")
                                ans = item.get("answers", {})
                                if isinstance(ans, dict) and "text" in ans and len(ans["text"]) > 0:
                                    ref_output = ans["text"][0]
                                else:
                                    ref_output = "No answer"
                            elif task == "sentiment":
                                input_text = item.get("text", "")
                                label = item.get("label", 0)
                                ref_output = "Positive" if label == 1 else ("Negative" if label == 0 else "Neutral")

                        # Clean
                        input_text = preprocessor.clean_text(str(input_text))
                        ref_output = preprocessor.clean_text(str(ref_output))
                        
                        # Deduplicate
                        if input_text in seen_inputs or not input_text:
                            continue
                        seen_inputs.add(input_text)
                        
                        sample_id = f"{task}_{lang}_{idx:05d}"
                        
                        processed = {
                            "id": sample_id,
                            "language": lang,
                            "task": task,
                            "input_text": input_text,
                            "reference_output": ref_output
                        }
                        processed_data.append(processed)
                        
                    except Exception as e:
                        log.debug(f"Error processing {task} {lang} line {idx}: {e}")
            
            with open(out_file, "w", encoding="utf-8") as f:
                for item in processed_data:
                    f.write(json.dumps(item) + "\n")
            
            log.info(f"Built structured dataset {out_file} with {len(processed_data)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()
    build_datasets(args.config, args.raw_dir, args.output_dir)
