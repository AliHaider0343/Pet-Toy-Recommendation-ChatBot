import openai
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]=api_key


#function to load documents 
def load_documents(dirctory="E:/COURSES/MACHINE_LEARNING/MAKTEK.IO/TOY_AI/implementation/PetToys.csv"):
 loader=CSVLoader(dirctory)
 documents = loader.load()
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
 docs = text_splitter.split_documents(documents)
 return docs

embedding_function=OpenAIEmbeddings()

#to store embeddings in vectordatabase 
chroma_db = Chroma.from_documents(
    documents = load_documents(),
    embedding = embedding_function,
    persist_directory="./chroma_db1"
    )



 



 
           





