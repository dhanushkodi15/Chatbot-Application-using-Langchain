from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
load_dotenv()

# text_loader = TextLoader("sample.txt")
# text_document = text_loader.load()
# print(text_document)

# web_loader = WebBaseLoader(web_path="https://docs.langchain.com/oss/python/langchain/overview")
# web_document = web_loader.load()
# print(web_document)

pdf_loader = PyPDFLoader("DA.pdf")
pdf_document = pdf_loader.load()

## Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splited_documents = text_splitter.split_documents(pdf_document)

## Embeddings
vector_db = Chroma.from_documents(splited_documents[:20],OpenAIEmbeddings())

if __name__ == "__main__":
    query = "tree from preorder"
    result = vector_db.similarity_search(query)
    print(result[0].page_content)


