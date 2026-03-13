import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from src.utils.logging import log

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Safely string to torch dtype.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)

class ModelLoader:
    def __init__(self, model_config: dict):
        """
        Expects a config dict like:
        {
            "hf_path": "meta-llama/Llama-3-8B-Instruct",
            "dtype": "bfloat16",
            "device_map": "auto"
        }
        """
        self.hf_path = model_config.get("hf_path")
        self.device_map = model_config.get("device_map", "auto")
        self.dtype_str = model_config.get("dtype", "float32")
        self.torch_dtype = get_torch_dtype(self.dtype_str)

        if not self.hf_path:
            raise ValueError("hf_path must be provided in model configuration.")

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads the HuggingFace model and tokenizer dynamically.
        """
        log.info(f"Loading tokenizer from: {self.hf_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_path, 
            trust_remote_code=True, 
            padding_side="left"  # Causal LM best practice for batched generation
        )
        
        # If model doesn't have a pad token, set it to EOS to prevent errors during batching
        if getattr(tokenizer, "pad_token", None) is None:
             tokenizer.pad_token = getattr(tokenizer, "eos_token", "<pad>")
             tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

        log.info(f"Loading Model: {self.hf_path} | Dtype: {self.torch_dtype} | Device Map: {self.device_map}")
        
        # Use torch.device explicitly for CPU environments (like our local testing setup without GPUs)
        if not torch.cuda.is_available() and self.device_map == "auto":
            log.warning("CUDA is not available. Overriding device_map to CPU fallback to prevent errors.")
            self.device_map = "cpu"

        try:
             # Create kwargs dict to cleanly omit device_map if it's 'cpu' and throws HF errors sometimes
             model_kwargs = {
                 "torch_dtype": self.torch_dtype,
                 "trust_remote_code": True,
                 "low_cpu_mem_usage": True
             }
             
             if torch.cuda.is_available():
                 model_kwargs["device_map"] = self.device_map
                 
             model = AutoModelForCausalLM.from_pretrained(
                 self.hf_path,
                 **model_kwargs
             )
             
             log.info("Model loaded successfully.")
             return model, tokenizer

        except Exception as e:
             log.error(f"Critical error loading model {self.hf_path}: {e}")
             raise e
