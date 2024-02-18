from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
# from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "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"
from app import file_path
# location of the pdf file/files.
pdf_path = r"file_path"
reader = PdfReader(pdf_path)

reader
# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
#raw text
raw_text[:100]

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
len(texts)
texts[0]
texts[1]

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)
docsearch

from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain_openai import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")
query = "You are a bot that creates notes. Create a detailed note of the lecture pdf that has been passed to you. Include Headings and subheadings with bulletpoints wherever necessary. Make sure to give detailed descriptoins. Add a small summary in the end. Also add a title to the page."

docs = docsearch.similarity_search(query)
# print(docs)

output = chain.run(input_documents=docs, question=query)
# output=chain.invoke({"question":"query","input_documents":"docs"})

import openpyxl
f = open('output.txt', "w")

f.writelines(output)
f.close()