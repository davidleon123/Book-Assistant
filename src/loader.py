from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

persist_directory = '../data'
vectordb = Chroma(
    persist_directory=persist_directory,
#    embedding_function=embedding
)


def load_data(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(docs)
    print(docs[0].page_content[0:100])
    return docs








#embedding = OpenAIEmbeddings()



def main():
    file_path = "../project/irs.pdf"
    load_data(file_path)
    #print(vectordb._collection.count())

if __name__ == "__main__":
    main()