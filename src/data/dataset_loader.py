import os
from datasets import load_dataset, Dataset
from src.utils.logging import log

def load_benchmark_dataset(data_dir: str, task: str, language: str) -> Dataset | None:
    """
    Load a preprocessed dataset for a specific task and language.
    Returns a HuggingFace Dataset object.
    """
    file_path = os.path.join(data_dir, f"{task}_{language}.jsonl")
    
    if not os.path.exists(file_path):
        log.error(f"Dataset file not found: {file_path}")
        return None
        
    try:
        ds = load_dataset("json", data_files=file_path, split="train")
        log.info(f"Loaded {len(ds)} samples for task: {task}, language: {language}")
        return ds
    except Exception as e:
        log.error(f"Failed to load dataset {file_path}: {e}")
        return None

def filter_dataset(dataset: Dataset, max_samples: int = None) -> Dataset:
    """
    Filter the dataset to a maximum number of samples.
    """
    if max_samples and max_samples < len(dataset):
        return dataset.select(range(max_samples))
    return dataset

class BenchmarkDataLoader:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir

    def get_task_data(self, task: str, language: str, limit: int = None) -> Dataset | None:
        """
        Retrieves a dataset ready for inference/evaluation.
        """
        ds = load_benchmark_dataset(self.data_dir, task, language)
        if ds and limit:
            ds = filter_dataset(ds, limit)
        return ds
