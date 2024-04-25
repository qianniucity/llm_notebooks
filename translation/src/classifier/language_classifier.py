import yaml
from transformers import pipeline

from base.config import Config


class LanguageDetector(Config):
    """XLMRoBERTa for language detection"""

    def __init__(self) -> None:
        super().__init__()
        self.model = pipeline(
            "text-classification",
            model=self.config["classifier"]["model"],
            device=self.config["classifier"]["device"],
        )

    def detect_language(self, document: str) -> str:
        """
        Detects the language of a document
        Args:
            document (str): document for language detection
        Returns:
            str: language
        """

        lang = self.model([document])[0]["label"]

        return self.lang_map[lang]
