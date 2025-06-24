from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
import os
import API_KEYS as keys


os.environ["GROQ_API_KEY"] = keys.GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"]=keys.LANGCHAIN_API_KEY

llama = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1
)


template = ChatPromptTemplate(
    messages=[
        ("system","you are a helpful and intelligent assistant and you wil answer all the queries of the user intelligently and accurately and giving as much informations as possible"),
        ("human","{input}")
    ]
)

parser= StrOutputParser()

chain = template | llama | parser 


def get_answer():
    question = input("Enter your question: ")
    output = chain.invoke({"input": question})
    print("\nAnswer:", output)


get_answer()
