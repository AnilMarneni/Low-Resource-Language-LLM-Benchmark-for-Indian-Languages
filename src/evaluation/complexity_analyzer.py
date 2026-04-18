import numpy as np
import stanza
from src.utils.logging import log

class ComplexityAnalyzer:
    def __init__(self, languages: list[str]):
        """
        Initialize the complexity analyzer with supported languages.
        """
        self.languages = languages
        self.pipelines = {}
        
        # Mapping of ISO codes to Stanza language codes
        self.lang_map = {
            "te": "te", # Telugu
            "kn": "kn", # Kannada
            "mr": "mr", # Marathi
            "ta": "ta", # Tamil
            "ml": "ml", # Malayalam
            "hi": "hi"  # Hindi (often useful as a baseline)
        }
        
    def _get_pipeline(self, lang: str):
        if lang not in self.lang_map:
            return None
            
        stanza_lang = self.lang_map[lang]
        if stanza_lang not in self.pipelines:
            try:
                # Initialize stanza pipeline for tokenization and sentence splitting
                # We use a lean processor list to keep it fast
                log.info(f"Loading Stanza pipeline for {stanza_lang}...")
                stanza.download(stanza_lang, processors='tokenize', quiet=True)
                self.pipelines[stanza_lang] = stanza.Pipeline(lang=stanza_lang, processors='tokenize', verbose=False)
            except Exception as e:
                log.warning(f"Could not load Stanza for {lang}: {e}")
                return None
        return self.pipelines[stanza_lang]

    def analyze_samples(self, texts: list[str], lang: str) -> dict:
        """
        Analyze a list of texts and return aggregated complexity metrics.
        """
        if not texts:
            return {}
            
        nlp = self._get_pipeline(lang)
        
        sentence_lengths = []
        token_lengths = []
        all_tokens = []
        
        for text in texts:
            if not text or not text.strip():
                continue
                
            if nlp:
                doc = nlp(text)
                sentences = doc.sentences
                num_sentences = len(sentences)
                tokens = [t.text for s in sentences for t in s.tokens]
            else:
                # Fallback to simple split
                sentences = text.split('.') # Very crude
                num_sentences = max(len(sentences), 1)
                tokens = text.split()
            
            num_tokens = len(tokens)
            if num_tokens > 0:
                sentence_lengths.append(num_tokens / num_sentences)
                token_lengths.extend([len(t) for t in tokens])
                all_tokens.extend([t.lower() for t in tokens])

        # Calculate Metrics
        metrics = {
            "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            "avg_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
            "lexical_diversity_ttr": self._calculate_ttr(all_tokens),
            "total_tokens_analyzed": len(all_tokens)
        }
        
        return metrics

    def _calculate_ttr(self, tokens: list[str]) -> float:
        """
        Calculate Type-Token Ratio (TTR) as a measure of lexical diversity.
        """
        if not tokens:
            return 0.0
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)

    def get_sample_complexity(self, text: str, lang: str) -> dict:
        """
        Get complexity metrics for a single sample.
        """
        return self.analyze_samples([text], lang)
