from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from book_assistant.ai_module.logger import log_question
from book_assistant.ai_module.config import BASE_DIR


if TYPE_CHECKING:
    from langchain_core.documents import Document
    

db_name = "programming_DB"
persist_directory = BASE_DIR / db_name  # where to store the database

EMBEDDING = OpenAIEmbeddings()

VECTORDB = Chroma(
    persist_directory=str(persist_directory),
    embedding_function=EMBEDDING
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=100
)


# Build prompt
template = """You are teaching programming. Use the following pieces of context to answer the question at the end.
 If no context is provided, just say that you don't know, don't try to make up an answer. 
 Use three sentences maximum. 
 Keep the answer as concise as possible. Use no more than 70 words and say how many words you used.
 Always give a positive word of encouragement to the student after your answer, extolling
 the virtue of educating oneself or of how exciting it is to be a programmer. 
 Give this encouragement without preceding with the mention `encouragement:` and limit 
 this encouragement to less than 8 words.
 
 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

QA_CHAIN = RetrievalQA.from_chain_type(
    llm,
    retriever=VECTORDB.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


def _load_data(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def _split_data(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)
    return splits


def _add_documents(documents: List[Document]) -> List[str]:
    ids = VECTORDB.add_documents(documents)
    return ids


def load(book_name: str) -> None:
    doc_to_load_path = BASE_DIR / "book_upload" / book_name
    docs = _load_data(str(doc_to_load_path))
    print(len(docs))
    print(docs[0].page_content[0:10])
    splits = _split_data(docs)
    print(len(splits))
    print(splits[0].page_content[0:10])
    ids = _add_documents(splits)
    print(len(ids))
    print(VECTORDB._collection.count())


DEBUG_ANSWER_PATH = BASE_DIR / "debug_answer.pkl"


def answer_question(question: str) -> Dict[str, Any]:
    result = QA_CHAIN.invoke({"query": question})
    return result


def save_answer(result: Dict[str, Any], file_path: str = str(DEBUG_ANSWER_PATH)) -> None:
    import pickle
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)


def load_answer(file_path: str = str(DEBUG_ANSWER_PATH)) -> Dict[str, Any]: 
    import pickle
    with open(file_path, 'rb') as file:
        result = pickle.load(file)
    return result


def format_answer(result: Dict[str, Any], max_source_documents: int = 5, max_num_words: int = 25) -> str:
    output = []
    output.append("The answer to the question is:")    
    output.append(result["result"])
    output.append("*" * 50)
    for i, doc in enumerate(result['source_documents'][:max_source_documents], start=1):
        output.append(f"Source document {i}:")
        output.append(Path(doc.metadata['source']).name)
        output.append("on page:")
        output.append(str(doc.metadata['page']))
        output.append("The first 25 words of the source are:")
        first_25_words = ' '.join(doc.page_content.split()[:max_num_words])
        output.append(first_25_words)
    
    return "\n".join(output)


def format_answer_django(result: Dict[str, Any], 
                         max_source_documents: int = 5,
                         max_num_words: int = 25) -> Dict[str, Any]:
    output = {"result": result["result"], "source_documents": []}
    
    for doc in result['source_documents'][:max_source_documents]:
        doc_info = {
            'source': Path(doc.metadata['source']).name,
            'page': str(doc.metadata['page']),
            'first_25_words': ' '.join(doc.page_content.split()[:max_num_words])
        }
        output['source_documents'].append(doc_info)
        
    return output


def display_answer(result: Dict[str, Any]) -> None:
    print(format_answer(result))


questions = ["what is a javascript closure?",
            "how do I create a timestamp with javascript?",
            "what is a list in javascript",
             "how to get started with javascript?",
             "how to use CSS with javascript?",
             "how to reverse an array?",
             ]
question = questions[1]


def main() -> None:
    print(question)
    log_question(question)
    answer = answer_question(question)
    display_answer(answer)
    save_answer(answer)
    # saved_answer = load_answer()
    # display_answer(saved_answer)
    

if __name__ == "__main__":
    main()