import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from base.config import Config

load_dotenv(f"{Path().parent.absolute()}/env/connection.env")


class VectorDatabase(Config):
    """PGVector database"""

    def __init__(self, encoder: HuggingFaceEmbeddings) -> None:
        """
        Init function
        Args:
            encoder (HuggingFaceEmbeddings): encoder to convert documents into embeddings
        """
        super().__init__()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config["retriever"]["passage"]["chunk_size"],
            chunk_overlap=self.config["retriever"]["passage"]["chunk_overlap"],
        )
        self.encoder = encoder
        self.conn_str = PGVector.connection_string_from_db_params(
            driver=os.getenv("DRIVER"),
            host=os.getenv("HOST"),
            port=os.getenv("PORT"),
            database=os.getenv("DATABASE"),
            user=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
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

    def store_passages_db(self, passages: list, id: str) -> None:
        """
        Store passages in vector database in embedding format
        Args:
            passages (list): list of passages
            id (str): id to identify the use case
        """
        PGVector.from_documents(
            embedding=self.encoder,
            documents=passages,
            collection_name=id,
            connection_string=self.conn_str,
            pre_delete_collection=True,
        )

    def retrieve_most_similar_document(self, question: str, k: int, id: str) -> str:
        """
        Retrieves the most similar document for a certain question
        Args:
            question (str): user question
            k (int): number of documents to query
            id (str): id to identify the use case
        Returns:
            str: most similar document
        """
        self.db = PGVector(
            collection_name=id,
            connection_string=self.conn_str,
            embedding_function=self.encoder,
        )
        docs = self.db.similarity_search(question, k=k)
        docs = [d.page_content for d in docs]

        return docs
