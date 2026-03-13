import os
import sys
import torch
from src.utils.logging import log
from src.utils.config_loader import load_yaml_config
from src.models.model_loader import ModelLoader
from src.models.inference_engine import InferenceEngine

def test_inference_pipeline():
    log.info("Starting Phase 3 verification: Model Loading and Inference")

    # Override for testing to avoid downloading massive models locally without GPUs
    test_config = {
        "hf_path": "HuggingFaceM4/tiny-random-LlamaForCausalLM", # tiny 1MB model for fast tests
        "dtype": "float32",
        "device_map": "cpu"
    }
    
    tasks_config = {
        "summarization": {
            "type": "generation",
            "max_new_tokens": 10,
            "prompt_template": "Summarize:\n\n{text}\n\nSummary:"
        }
    }

    try:
        log.info("Testing ModelLoader...")
        loader = ModelLoader(test_config)
        model, tokenizer = loader.load_model_and_tokenizer()
        
        assert model is not None, "Model loaded as None!"
        assert tokenizer is not None, "Tokenizer loaded as None!"
        
        log.info("Testing InferenceEngine initialization...")
        engine = InferenceEngine(model, tokenizer, batch_size=2)
        
        log.info("Testing Prompt Formatting...")
        sample = {
            "id": "test_01",
            "language": "te",
            "task": "summarization",
            "input_text": "This is a long Telugu article about artificial intelligence. It explains how models work.",
            "reference_output": "AI models are cool."
        }
        formatted = engine.format_prompt(tasks_config["summarization"]["prompt_template"], "summarization", sample)
        assert "Summarize:" in formatted, "Prompt formatting failed to template correctly."
        assert "artificial intelligence" in formatted, "Prompt formatting failed to include text."

        log.info("Testing Generation Logic...")
        # Mock batch of 2
        batch = [formatted, formatted]
        predictions = engine.generate_batch(batch, max_new_tokens=5)
        
        assert len(predictions) == 2, "Batch generation did not return correct number of predictions."
        assert isinstance(predictions[0], str), "Prediction output is not a string."
        
        log.info(f"Sample generation success! Output: {predictions[0]}")
        log.info("Phase 3 integration tests PASSED successfully.")
        
    except Exception as e:
        log.error(f"Phase 3 verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_inference_pipeline()
