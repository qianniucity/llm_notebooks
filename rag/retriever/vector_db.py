from base.config import Config
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


class VectorDatabase(Config):
    """FAISS database"""

    def __init__(self) -> None:
        super().__init__()
        self.retriever = FAISS
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config["retriever"]["passage"]["chunk_size"],
            chunk_overlap=self.config["retriever"]["passage"]["chunk_overlap"],
        )

    def create_passages_from_documents(self, documents: list) -> list:
        """
        Splits the documents into passages of a certain length
        Args:
            documents (list): list of documents
        Returns:
            list: list of passages
        """
        return self.text_splitter.split_documents(documents)

    def store_passages_db(self, passages: list, encoder: HuggingFaceEmbeddings) -> None:
        """
        Store passages in vector database in embedding format
        Args:
            passages (list): list of passages
            encoder (HuggingFaceEmbeddings): encoder to convert passages into embeddings
        """
        self.db = self.retriever.from_documents(passages, encoder)

    def retrieve_most_similar_document(self, question: str, k: int) -> str:
        """
        Retrieves the most similar document for a certain question
        Args:
            question (str): user question
            k (int): number of documents to query
        Returns:
            str: most similar document
        """

        docs = self.db.similarity_search(question, k=k)
        docs = [d.page_content for d in docs]

        return '\n'.join(docs)
