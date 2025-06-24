from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llama = Ollama(
    model="llama3.2:latest",
    temperature=0.1
)


template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and intelligent assistant. Answer all user queries intelligently and with as much detail as possible."),
    ("human", "{input}")
])

parser = StrOutputParser()


chain = template | llama | parser

def get_answer():
    question = input("Enter your question: ")
    output = chain.invoke({"input": question})
    print("\nAnswer:", output)

get_answer()
