from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI

import os
from dotenv import load_dotenv
from langserve import add_routes
import uvicorn
load_dotenv()

app = FastAPI(
    title="Langchain ChatBot API",
    version="1.0.0",
    description="This is a simple API for Langchain ChatBot using FastAPI and Langchain"
)

prompt1 = ChatPromptTemplate.from_template("{prompt}")
prompt2 = ChatPromptTemplate.from_template("{prompt}")

gemini = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
groq = ChatGroq(model="llama-3.3-70b-versatile")

add_routes(
    app, prompt1|gemini, path="/googlegenai"
)

add_routes(
    app, prompt2|groq, path = "/groq"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)



