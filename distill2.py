# -*- coding: utf-8 -*-


import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import textwrap

wrapper = textwrap.TextWrapper(width=80,
    initial_indent=" " * 8,
    subsequent_indent=" " * 8,
    break_long_words=False,
    break_on_hyphens=False)

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from optimum.bettertransformer import BetterTransformer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True#, use_flash_attention_2=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15, #long form transcription
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)



result = pipe('Worcester Hall.mp3')
#print(result["text"])

f = open('lecture_transcript.txt','w')
f.writelines(result["text"])
f.close()

"""Now, We will treat our transcript by making chunks for langchain to parse

Importing all Langchain Stuff
"""

from PyPDF2 import PdfReader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone,FAISS
from langchain_community.vectorstores import Weaviate
from transformers import GPT2TokenizerFast
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
#from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "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"

"""Now we will create chunks of the data stored in our transcript."""

f = open('lecture_transcript.txt','r')
text = f.read()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap = 24,
    #length_function= count_tokens,
)
chunks = text_splitter.create_documents([text])

"""Now, we will work on the Embeddings"""

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(text, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
#from langchain.chains.combine_documents import DocumentLoadersChain
from langchain.llms import OpenAI

"""Now, we will work on Summarizing the Text"""

f = open('lecture_transcript.txt','r')
text = f.read()

template = '''You are a bot that creates notes. Create a detailed note of the lecture pdf that has been passed to you. Include Headings and subheadings with bulletpoints wherever necessary. Make sure to give detailed descriptoins. Add a small summary in the end. `{text}`
'''
prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

llm=ChatOpenAI(model_name='gpt-3.5-turbo')

from langchain.chains.summarize import load_summarize_chain

chunks_prompt="""
Please summarize in detail the below lecture along with explanation in atleast 2500 characters:
Lecture:`{text}'
Summary:
"""
map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt)

final_combine_prompt='''
You are a bot that creates notes. Create a detailed note of the lecture pdf that has been passed to you. Include Headings and bulletpoints wherever necessary. Make sure to give detailed descriptoins. Add a small summary in the end.

Provide a final summary of the entire lecture with these important points and their explanations.

Start with introducing the idea,

Start the summary with an introduction and provide the necessary explanation thoroughly in bullet points



End with conclusion
Lecture: `{text}`
'''
final_combine_prompt_template=PromptTemplate(input_variables=['text'],
                                             template=final_combine_prompt)

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template,
    combine_prompt=final_combine_prompt_template,
    verbose=False
)
output = summary_chain.run(chunks)
f = open('output.txt', "w")


f.writelines(output)
f.close()

"""Lastly, we will put some questions inside our output file."""

#query = "Objective: Summarize the key points and main concepts discussed in the lecture recording in a structured format with bullet points, headings, and subheadings whereever necessary. Begin with an introduction providing a brief overview of the lecture's topic and importance. Then, identify and summarize the main points covered in the lecture, followed by breaking down the main points into subtopics, if applicable, and summarizing each subtopic separately. Highlight any important concepts, theories, or ideas discussed, and provide examples or case studies to illustrate key points. Conclude by summarizing the main takeaways from the lecture and reiterating its significance.Please generate a well-structured summary of the lecture recording based on the provided prompt. Ensure that the summary includes clear headings, subheadings, and bullet points to organize the information effectively. Make it look presentable and formatted"
#query = ""'
query = "Give me 10 questions for a quiz from the lecture that covers important matter"
#query = "You are a bot that creates notes. Create a detailed note of the lecture pdf that has been passed to you. Include Headings and subheadings with bulletpoints wherever necessary. Make sure to give detailed descriptoins. Add a small summary in the end. "
#query = "Summarize the entire document untill lecture 6 in detail"
docs = docsearch.similarity_search(query)
results = chain.run(input_documents=docs, question=query)

import openpyxl
f = open('output.txt', "a")


f.writelines(results)
f.close()