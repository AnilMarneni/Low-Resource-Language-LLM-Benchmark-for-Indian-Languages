import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from src.utils.logging import log

class InferenceEngine:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = self.model.device
        
        # Ensure model is in eval mode
        self.model.eval()

    def format_prompt(self, template: str, task: str, sample: dict) -> str:
        """
        Formats the input text using the task-specific prompt template.
        """
        if task == "qa":
            # Splitting context and question (we joined them in preprocessing for basic schema)
            parts = sample["input_text"].split("?")
            context = parts[0] if len(parts) > 1 else sample["input_text"]
            question = parts[1] + "?" if len(parts) > 1 else "What is the answer?"
            return template.format(context=context, question=question)
        
        elif task == "translation":
            return template.format(target_lang=sample["language"], text=sample["input_text"])
            
        else:
            return template.format(text=sample["input_text"])

    @torch.no_grad()
    def generate_batch(self, batched_prompts: list[str], max_new_tokens: int) -> list[str]:
        """
        Runs generation on a batch of formatted prompts.
        """
        inputs = self.tokenizer(
            batched_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Calculate prompt lengths to extract only the generated response
        prompt_lengths = [len(p) for p in inputs.input_ids]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False, # Greedy decoding for benchmarking reproducibility
            use_cache=True
        )

        # Decode only the newly generated tokens
        generated_texts = []
        for i, output in enumerate(outputs):
            # Extract just the newly generated tokens
            new_tokens = output[prompt_lengths[i]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_texts.append(text)

        return generated_texts

    def run_inference(self, dataset: Dataset, task: str, config: dict, output_dir: str = "results/predictions"):
        """
        Runs inference over a full dataset and saves predictions to JSONL.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        tasks_config = config.get("tasks", {})
        task_info = tasks_config.get(task, {})
        prompt_template = task_info.get("prompt_template", "{text}")
        max_new_tokens = task_info.get("max_new_tokens", 128)
        
        lang = dataset[0]["language"]
        output_file = os.path.join(output_dir, f"{task}_{lang}_predictions.jsonl")
        
        log.info(f"Starting inference for {task} ({lang}) -> {output_file}")
        
        # DataLoader for batching
        def collate_fn(batch):
            return batch
            
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        
        results = []
        
        for batch in tqdm(dataloader, desc=f"Inference [{task} - {lang}]"):
            prompts = [self.format_prompt(prompt_template, task, sample) for sample in batch]
            
            try:
                predictions = self.generate_batch(prompts, max_new_tokens)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.error(f"OOM Error during batch generation. Try lowering batch size. Error: {e}")
                    # Recovery strategy: clear cache and try one-by-one
                    torch.cuda.empty_cache()
                    predictions = []
                    for prompt in prompts:
                        predictions.extend(self.generate_batch([prompt], max_new_tokens))
                else:
                    raise e
                    
            for i, sample in enumerate(batch):
                results.append({
                    "id": sample["id"],
                    "language": sample["language"],
                    "task": sample["task"],
                    "input_text": sample["input_text"],
                    "reference_output": sample["reference_output"],
                    "prediction": predictions[i]
                })

        # Save to disk
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
                
        log.info(f"Saved {len(results)} predictions to {output_file}")
        return output_file
