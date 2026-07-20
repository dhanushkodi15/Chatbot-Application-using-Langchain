from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
prompt_template = ChatPromptTemplate()

#Wikipedia content wrapper
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

#Arxiv research website wrapper
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#Document wrapper from website
web_loader = WebBaseLoader(web_path='https://roadmap.sh/ai-engineer')
web_docs = web_loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(web_docs)
vector_db = FAISS.from_documents(documents,OpenAIEmbeddings())

#retrival tool creation for 
retriever = vector_db.as_retriver()
retriver_tool = create_retriever_tool(retriever,"AI Engineering RoadMap","Search when it ask about AI engineering roadmap")

#Create agent tool for three wrappers
tools = [wiki,arxiv,retriver_tool]

agent_tool = create_openai_tools_agent(llm,tools,prompt_template)
agent_executer = AgentExecutor(agent=agent_tool,tools=tools)

agent_executer.invoke({"input":"Give AI engineering RoadMap"})



