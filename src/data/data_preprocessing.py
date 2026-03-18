import re
import html
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import sentence_tokenize
    from indicnlp.tokenize import indic_tokenize
except ImportError:
    IndicNormalizerFactory = None

from src.utils.logging import log

class DataPreprocessor:
    def __init__(self, language="te"):
        self.language = language
        # Language codes for IndicNLP mapping
        self.indic_lang_map = {
            "te": "te",
            "kn": "kn",
            "mr": "mr",
            "ta": "ta",
            "hi": "hi",
            "bn": "bn",
            "gu": "gu",
            "pa": "pa",
            "ml": "ml"
        }
        self.indic_lang = self.indic_lang_map.get(language, "en")
        
        self.normalizer = None
        if IndicNormalizerFactory and self.indic_lang in ["te", "kn", "mr", "ta", "hi", "bn"]:
            try:
                factory = IndicNormalizerFactory()
                self.normalizer = factory.get_normalizer(self.indic_lang)
            except Exception as e:
                log.warning(f"Could not load IndicNormalizer for {self.indic_lang}: {e}")

    def remove_html(self, text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_unicode(self, text: str) -> str:
        if self.normalizer:
            try:
                text = self.normalizer.normalize(text)
            except Exception as e:
                log.warning(f"Error normalizing unicode for {self.indic_lang}: {e}")
        return text

    def segment_sentences(self, text: str) -> list[str]:
        if sentence_tokenize and self.indic_lang in self.indic_lang_map.values():
            try:
                return sentence_tokenize.sentence_split(text, lang=self.indic_lang)
            except Exception as e:
                log.warning(f"Error segmenting sentences for {self.indic_lang}: {e}")
        # Fallback simple split
        return text.split('।') if '।' in text else text.split('.')

    def tokenize(self, text: str) -> list[str]:
        if indic_tokenize and self.indic_lang in self.indic_lang_map.values():
            try:
                return indic_tokenize.trivial_tokenize(text)
            except Exception as e:
                log.warning(f"Error tokenizing for {self.indic_lang}: {e}")
        return text.split()

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = self.remove_html(text)
        text = self.normalize_unicode(text)
        return text.strip()
