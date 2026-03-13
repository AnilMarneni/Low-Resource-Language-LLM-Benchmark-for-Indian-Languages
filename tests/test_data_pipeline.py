import os
import sys
from src.utils.logging import log
from src.utils.config_loader import load_yaml_config
from src.data.dataset_loader import BenchmarkDataLoader

def test_pipeline():
    log.info("Starting comprehensive Phase 1 & 2 verification...")
    
    # 1. Test Configurations
    log.info("Testing Configuration Loading...")
    try:
        benchmark_config = load_yaml_config("configs/benchmark.yaml")
        tasks_config = load_yaml_config("configs/tasks.yaml")
        models_config = load_yaml_config("configs/models.yaml")
        
        assert "languages" in benchmark_config["benchmark"], "Missing languages in benchmark configs."
        assert "te" in benchmark_config["benchmark"]["languages"], "Missing 'te' language config."
        assert "llama_3_8b_instruct" in models_config["models"], "Missing LLaMA benchmark config."
        assert "summarization" in tasks_config["tasks"], "Missing valid summarization task setup."
        
        log.info("Configuration checks passed!")
    except Exception as e:
        log.error(f"Configuration test failed: {e}")
        sys.exit(1)

    # 2. Test Data Generation Outputs
    log.info("Testing Raw Data Generation...")
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
         log.error(f"Raw data directory missing at {raw_dir}")
         sys.exit(1)
         
    raw_files = os.listdir(raw_dir)
    assert len(raw_files) > 0, "No raw data files generated!"
    log.info(f"Found {len(raw_files)} raw data files.")
    
    # 3. Test Processed Pipeline schemas
    log.info("Testing Processed Data pipeline via Loader...")
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
         log.error(f"Processed data directory missing at {processed_dir}")
         sys.exit(1)
         
    processed_files = os.listdir(processed_dir)
    assert len(processed_files) > 0, "No datasets built successfully!"
    
    loader = BenchmarkDataLoader(data_dir=processed_dir)
    test_task = "summarization"
    test_lang = "te"
    
    try:
        ds = loader.get_task_data(test_task, test_lang, limit=10)
        assert ds is not None, f"Dataset loader returned None for {test_task}_{test_lang}"
        assert len(ds) == 10, "Loader 'limit' functionality failed."
        
        sample = ds[0]
        expected_keys = {"id", "language", "task", "input_text", "reference_output"}
        assert expected_keys.issubset(sample.keys()), f"Missing keys in dataset schema. Found: {sample.keys()}"
        
        assert sample["language"] == test_lang, "Language mismatch in struct"
        assert sample["task"] == test_task, "Task mismatch in struct"
        assert isinstance(sample["input_text"], str), "Input text is not a string!"
        assert isinstance(sample["reference_output"], str), "Reference output is not a string!"
        
        log.info("Dataset Loading structure checks PASSED!")
    except Exception as e:
        log.error(f"Dataset Pipeline checks failed: {e}")
        sys.exit(1)

    log.info("All Phase 1 & 2 integration tests PASSED successfully.")

if __name__ == "__main__":
    test_pipeline()
