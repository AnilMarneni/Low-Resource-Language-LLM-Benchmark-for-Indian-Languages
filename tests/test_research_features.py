import pytest
from src.evaluation.metrics import compute_metrics
from src.evaluation.complexity_analyzer import ComplexityAnalyzer

def test_advanced_metrics():
    predictions = ["The cat is on the mat.", "I love coding."]
    references = ["The cat is sitting on the mat.", "I enjoy programming."]
    
    # Test with semantic metrics
    results = compute_metrics("summarization", predictions, references, lang="en")
    
    assert "bert_score" in results
    assert results["bert_score"] > 0
    assert "rougeL" in results

def test_translation_metrics():
    predictions = ["नमस्ते, आप कैसे हैं?"]
    references = ["नमस्ते, आप कैसे हैं?"]
    results = compute_metrics("translation", predictions, references, lang="hi")
    
    assert "sacrebleu" in results
    assert "chrf++" in results
    assert results["sacrebleu"] > 99.0 # Exact match
    assert "bert_score" in results

def test_complexity_analyzer():
    analyzer = ComplexityAnalyzer(["te", "hi"])
    text = "This is a simple sentence. This is another one."
    
    metrics = analyzer.get_sample_complexity(text, "en") # English fallback
    
    assert metrics["avg_sentence_length"] == 4.5
    assert metrics["lexical_diversity_ttr"] > 0
    assert metrics["total_tokens_analyzed"] == 9

def test_indic_complexity():
    analyzer = ComplexityAnalyzer(["hi"])
    # "नमस्ते, आप कैसे हैं?" -> 4 tokens
    text = "नमस्ते, आप कैसे हैं?"
    metrics = analyzer.get_sample_complexity(text, "hi")
    
    assert metrics["total_tokens_analyzed"] > 0
    assert "avg_sentence_length" in metrics
