from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings


persist_directory = '../data'

#embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
#    embedding_function=embedding
)

print(vectordb._collection.count())