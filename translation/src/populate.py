import logging

import pandas as pd
from langchain.docstore.document import Document

from encoder.encoder import Encoder
from retriever.vector_db import VectorDatabase

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":

    logger.info("Init encoder and vectorDB classes..")
    encoder = Encoder()
    vectordb = VectorDatabase(encoder.encoder)

    logger.info("Loading data..")
    df = pd.read_csv("data/data.csv")

    logger.info("Creating full review column..")
    df["full_review"] = df[["reviews.title", "reviews.text"]].apply(
        lambda row: ". ".join(row.values.astype(str)), axis=1
    )

    logger.info("Storing reviews in vectorDB..")
    for product_id in df["asins"].unique()[:10]:
        # create documents to store in Postgres
        docs = [
            Document(page_content=item)
            for item in df[df["asins"] == product_id]["full_review"].tolist()
        ]

        passages = vectordb.create_passages_from_documents(docs)
        vectordb.store_passages_db(passages, product_id)
