import json
import os
from datetime import datetime
from src.utils.logging import log

class ProgressMonitor:
    def __init__(self, progress_file="results/live_progress.json"):
        self.progress_file = progress_file
        self.state = {
            "status": "idle",
            "current_model": "",
            "current_task": "",
            "current_lang": "",
            "progress_percent": 0.0,
            "batches_completed": 0,
            "total_batches": 0,
            "last_update": ""
        }
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        self._write_state()

    def update(self, **kwargs):
        """
        Update the progress state with provided key-value pairs.
        """
        self.state.update(kwargs)
        self.state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_state()

    def _write_state(self):
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            log.error(f"Failed to write progress monitor state: {e}")

    def complete(self):
        self.update(status="completed", progress_percent=100.0)
        log.info("Benchmark execution marked as completed in monitor.")

    def reset(self):
        self.state = {
            "status": "running",
            "current_model": "",
            "current_task": "",
            "current_lang": "",
            "progress_percent": 0.0,
            "batches_completed": 0,
            "total_batches": 0,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._write_state()
