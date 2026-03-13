import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.utils.logging import log

# Initialize evaluate modules (can be heavy, handle downloads gracefully)
try:
    rouge = evaluate.load("rouge")
    sacrebleu = evaluate.load("sacrebleu")
except Exception as e:
    log.warning(f"Could not load standard HF evaluate metrics immediately. Initializing inline. Error: {e}")
    rouge, sacrebleu = None, None

def compute_summarization_metrics(predictions: list[str], references: list[str]) -> dict:
    global rouge
    if not rouge:
        rouge = evaluate.load("rouge")
        
    # ROUGE expects list of strings
    try:
        results = rouge.compute(predictions=predictions, references=references)
        return {
            "rouge1": results.get("rouge1", 0.0),
            "rouge2": results.get("rouge2", 0.0),
            "rougeL": results.get("rougeL", 0.0)
        }
    except Exception as e:
        log.warning(f"Summarization metric error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def compute_translation_metrics(predictions: list[str], references: list[str]) -> dict:
    global sacrebleu
    if not sacrebleu:
        sacrebleu = evaluate.load("sacrebleu")
        
    # sacreBLEU expects a list of lists for references
    refs = [[r] for r in references]
    try:
        results = sacrebleu.compute(predictions=predictions, references=refs)
        return {
            "sacrebleu": results.get("score", 0.0)
        }
    except Exception as e:
        log.warning(f"Translation metric error: {e}")
        return {"sacrebleu": 0.0}

def compute_qa_metrics(predictions: list[str], references: list[str]) -> dict:
    # Exact Match and simple word-level F1
    exact_matches = 0
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        # Exact Match
        if pred.lower().strip() == ref.lower().strip():
            exact_matches += 1
            
        # F1
        common = pred_words.intersection(ref_words)
        if not common:
            f1_scores.append(0.0)
            continue
            
        precision = len(common) / len(pred_words)
        recall = len(common) / len(ref_words)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
        
    return {
        "exact_match": exact_matches / max(len(predictions), 1),
        "f1": np.mean(f1_scores) if f1_scores else 0.0
    }

def compute_sentiment_metrics(predictions: list[str], references: list[str]) -> dict:
    # Clean up preds to match refs strictly
    clean_preds = [p.strip().title() for p in predictions]
    clean_refs = [r.strip().title() for r in references]
    
    try:
        acc = accuracy_score(clean_refs, clean_preds)
        mac_f1 = f1_score(clean_refs, clean_preds, average='macro', zero_division=0)
    except Exception as e:
        log.warning(f"Error computing sentiment metrics: {e}")
        acc, mac_f1 = 0.0, 0.0
        
    return {
        "accuracy": float(acc),
        "macro_f1": float(mac_f1)
    }

def compute_metrics(task: str, predictions: list[str], references: list[str]) -> dict:
    if task == "summarization":
        return compute_summarization_metrics(predictions, references)
    elif task == "translation":
        return compute_translation_metrics(predictions, references)
    elif task == "qa":
        return compute_qa_metrics(predictions, references)
    elif task == "sentiment":
        return compute_sentiment_metrics(predictions, references)
    else:
        log.error(f"Unknown task {task} provided to metric computation.")
        return {}
