from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
import os

if TYPE_CHECKING:
    from langchain_core.documents import Document
    
load_dotenv(find_dotenv()) # read local .env file



project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_name = "data"
persist_directory = os.path.join(project_root, db_name) # where to store the database

EMBEDDING = OpenAIEmbeddings()

VECTORDB = Chroma(
    persist_directory=persist_directory,
    embedding_function=EMBEDDING
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=100
)


# Build prompt
template = """Use the following pieces of context to answer the question at the end.
 If you don't know the answer, just say that you don't know, don't try to make up an answer. 
 Use three sentences maximum. 
 Keep the answer as concise as possible. 
 Summarize the answer in less than 50 words.
 Always say "thanks for asking!" at the end of the answer. 
 Add at the end how many words you used to answer the question.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=VECTORDB.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


def _load_data(file_path: str)->List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def _split_data(documents: List[Document])->List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )
    splits = text_splitter.split_documents(documents)
    return splits

def _add_documents(documents: List[Document])->List[str]:
    ids = VECTORDB.add_documents(documents)
    return ids


def load()->None:
    doc_to_load_path = os.path.join(project_root, "irs.pdf")
    docs = _load_data(doc_to_load_path)
    print(len(docs))
    print(docs[0].page_content[0:10])
    splits = _split_data(docs)
    print(len(splits))
    print(splits[0].page_content[0:10])
    ids = _add_documents(splits)
    print(ids)
    print(VECTORDB._collection.count())

DEBUG_ANSWER_PATH = os.path.join(project_root, "debug_answer.pkl")


def answer_question(question: str) -> Dict[str, Any]:
    result = qa_chain.invoke({"query": question})
    return result

def save_answer(result: Dict[str, Any], file_path: str = DEBUG_ANSWER_PATH) -> None:
    import pickle
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)

def load_answer(file_path: str = DEBUG_ANSWER_PATH) -> Dict[str, Any]: 
    import pickle
    with open(file_path, 'rb') as file:
        result = pickle.load(file)
    return result

def display_answer(result: Dict[str, Any]) -> None:
    print("The answer to the question is:")    
    print(result["result"])
    print("The source for the answer is:")
    print(os.path.basename(result['source_documents'][0].metadata['source']))
    print("on page:")
    print(result['source_documents'][0].metadata['page'])
    print("The first 25 characters of the source are:")
    print(result['source_documents'][0].page_content[:25])

def main()->None:
    #load()
    question ="when do I have to declare  my taxes?"
    result = qa_chain({"query": question})
    display_answer(result)
    save_answer(result)
    saved_answer = load_answer()
    display_answer(saved_answer)
    
    # print("the source for the answer is:")
    # print(result["source_documents"][0])
    # print("The second source for the answer is:")
    # print(result["source_documents"][1], "...")

    #print(result["source_documents"][0])

    

if __name__ == "__main__":
    main()