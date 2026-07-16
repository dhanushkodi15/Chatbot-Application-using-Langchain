from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","Now act as a good assistant help to solve the problem"),
        ("user","Question:{question}")
    ]
)

# Streamlit Framework
st.title("Langchain ChatBot")
input_text = st.text_input("Hey! Whats on your mind!")

# LLm Connection
LLm = ChatOpenAI(model="gpt-3.5-turbo")
output = StrOutputParser()
chain = prompt_template | LLm | output

if input_text:
    st.write(chain.invoke({'question':input_text}))