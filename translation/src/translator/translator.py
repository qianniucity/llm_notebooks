from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

from base.config import Config


class Translator(Config):
    """mBART translator"""

    def __init__(self) -> None:
        super().__init__()
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.config["translator"]["model"]
        )
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.config["translator"]["model"], use_fast=False
        )

    def translate(self, document: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a given document based on the source and target language
        Args:
            document (str): document to translate
            source_lang (str): token for source language
            target_lang (str): token for target language
        Returns:
            str: translation
        """

        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(document, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
        )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ]
