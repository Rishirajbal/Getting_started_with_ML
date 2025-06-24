from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import os
from langchain_community.llms import Ollama
#from dotenv import load_dotenv
#load_dotenv(dotenv_path="API_KEYS.env")
#os.environ["GROQ_API_KEY"] = "your api here"
# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY") 
#initialize the model
#model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",temperature=0.1)
model=Ollama(model="llama3.2:latest",temperature=0.1)
#load and split 
loader=TextLoader("youtube_transcript.txt")
text=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
texts=text_splitter.split_documents(text)

#embed the text into Faiss
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db=FAISS.from_documents(texts,embeddings)

#prompt_design
prompt=ChatPromptTemplate.from_template(
    """Your are a intelligent,friendly and helpful AI assistant,solve each and every query of the user as the user is very busy and on a tight schedule,be very concise and precise in what u say and keep in mind the context {context} and answer the questions given in the input {input}"""
)

#retriever
simple_chain=create_stuff_documents_chain(model,prompt)
retriever=db.as_retriever()  
chain=create_retrieval_chain(retriever,simple_chain)

response = chain.invoke({"input": "What is agenda of the video?"})
print(response)
