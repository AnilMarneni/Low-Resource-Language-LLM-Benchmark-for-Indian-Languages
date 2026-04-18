import evaluate
import numpy as np
import torch
from bert_score import score as bert_score_compute
from sklearn.metrics import accuracy_score, f1_score
from src.utils.logging import log

# Initialize evaluate modules (can be heavy, handle downloads gracefully)
try:
    rouge = evaluate.load("rouge")
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
except Exception as e:
    log.warning(f"Could not load standard HF evaluate metrics immediately. Initializing inline. Error: {e}")
    rouge, sacrebleu, chrf = None, None, None

def compute_summarization_metrics(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    global rouge
    if not rouge:
        rouge = evaluate.load("rouge")
    
    results = {}
    try:
        rouge_res = rouge.compute(predictions=predictions, references=references)
        results.update({
            "rouge1": rouge_res.get("rouge1", 0.0),
            "rouge2": rouge_res.get("rouge2", 0.0),
            "rougeL": rouge_res.get("rougeL", 0.0)
        })
    except Exception as e:
        log.warning(f"Summarization ROUGE error: {e}")

    # Add BERTScore for Semantic Similarity
    try:
        # Use MuRIL for Indian Languages, otherwise default which usually uses RoBERTa/BERT
        model_type = "google/muril-base-cased" if lang in ["te", "kn", "hi", "mr", "ta", "ml", "bn", "gu", "pa", "or"] else None
        P, R, F1 = bert_score_compute(predictions, references, lang=lang, model_type=model_type, device="cuda" if torch.cuda.is_available() else "cpu")
        results["bert_score"] = float(F1.mean())
    except Exception as e:
        log.warning(f"BERTScore error: {e}")

    return results

def compute_translation_metrics(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    global sacrebleu, chrf
    if not sacrebleu:
        sacrebleu = evaluate.load("sacrebleu")
    if not chrf:
        chrf = evaluate.load("chrf")
        
    results = {}
    # sacreBLEU expects a list of lists for references
    refs = [[r] for r in references]
    try:
        sb_res = sacrebleu.compute(predictions=predictions, references=refs)
        results["sacrebleu"] = sb_res.get("score", 0.0)
    except Exception as e:
        log.warning(f"Translation sacrebleu error: {e}")

    try:
        chrf_res = chrf.compute(predictions=predictions, references=refs, word_order=2) # word_order=2 makes it chrF++
        results["chrf++"] = chrf_res.get("score", 0.0)
    except Exception as e:
        log.warning(f"Translation chrF++ error: {e}")

    # BERTScore for translation too
    try:
        model_type = "google/muril-base-cased" if lang in ["te", "kn", "hi", "mr", "ta", "ml", "bn", "gu", "pa", "or"] else None
        P, R, F1 = bert_score_compute(predictions, references, lang=lang, model_type=model_type, device="cuda" if torch.cuda.is_available() else "cpu")
        results["bert_score"] = float(F1.mean())
    except Exception as e:
        log.warning(f"BERTScore error: {e}")

    return results

def compute_qa_metrics(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    exact_matches = 0
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Improved tokenization for Indian languages could be done here, 
        # but for now we'll stick to simple word-level with better cleaning
        pred_words = set(pred.lower().strip().split())
        ref_words = set(ref.lower().strip().split())
        
        if pred.lower().strip() == ref.lower().strip():
            exact_matches += 1
            
        common = pred_words.intersection(ref_words)
        if not common:
            f1_scores.append(0.0)
            continue
            
        precision = len(common) / max(len(pred_words), 1)
        recall = len(common) / max(len(ref_words), 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
        f1_scores.append(f1)
        
    results = {
        "exact_match": exact_matches / max(len(predictions), 1),
        "f1": np.mean(f1_scores) if f1_scores else 0.0
    }

    # Add BERTScore for open-domain QA semantic similarity
    try:
        model_type = "google/muril-base-cased" if lang in ["te", "kn", "hi", "mr", "ta", "ml", "bn", "gu", "pa", "or"] else None
        P, R, F1 = bert_score_compute(predictions, references, lang=lang, model_type=model_type, device="cuda" if torch.cuda.is_available() else "cpu")
        results["bert_score"] = float(F1.mean())
    except Exception as e:
        log.warning(f"BERTScore error: {e}")

    return results

def compute_sentiment_metrics(predictions: list[str], references: list[str]) -> dict:
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

def compute_metrics(task: str, predictions: list[str], references: list[str], lang: str = "en") -> dict:
    if task == "summarization":
        return compute_summarization_metrics(predictions, references, lang)
    elif task == "translation":
        return compute_translation_metrics(predictions, references, lang)
    elif task == "qa":
        return compute_qa_metrics(predictions, references, lang)
    elif task == "sentiment":
        return compute_sentiment_metrics(predictions, references)
    else:
        log.error(f"Unknown task {task} provided to metric computation.")
        return {}
