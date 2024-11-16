from __future__ import annotations
from typing import TYPE_CHECKING, List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

import os

if TYPE_CHECKING:
    from langchain_core.documents import Document
    
load_dotenv(find_dotenv()) # read local .env file



project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_name = "data"
persist_directory = os.path.join(project_root, db_name) # where to store the database

EMBEDDING = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=EMBEDDING
)


def load_data(file_path: str)->List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_data(documents: List[Document])->List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )
    splits = text_splitter.split_documents(documents)
    return splits




def main()->None:
    doc_to_load_path = os.path.join(project_root, "irs.pdf")
    docs = load_data(doc_to_load_path)
    print(len(docs))
    print(docs[0].page_content[0:10])
    splits = split_data(docs)
    print(len(splits))
    print(splits[0].page_content[0:10])
    ids = vectordb.add_documents(splits)
    print(ids)

    print(vectordb._collection.count())

if __name__ == "__main__":
    main()