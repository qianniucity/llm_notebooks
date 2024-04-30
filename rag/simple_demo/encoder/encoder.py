from base.config import Config
from langchain.embeddings import HuggingFaceEmbeddings


class Encoder(Config):
    """Encoder to create workds embeddings from text"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = HuggingFaceEmbeddings(
            model_name=self.config["encoder"]["model_path"],
            model_kwargs=self.config["encoder"]["model_kwargs"],
            encode_kwargs=self.config["encoder"]["encode_kwargs"],
        )
