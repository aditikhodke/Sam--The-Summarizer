# Sam--The-Summarizer

## Overview
Our project involves the development of a custom GPT model tailored to summarizing lecture audios and PDF files. To achieve this, we utilized OpenAI's Whisper model for converting user-provided MP3 files into text format. Then, I implemented a process to segment the text file into chunks, which were then stored in a vector database to enable efficient searching through the LangChain framework. Additionally, I established a summarization chain employing map-reduce techniques, ensuring that each chunk of information was succinctly summarized to provide relevant outputs.


## Architecture

1. mp3 to text conversation
- OpenAI Whisper:

  * Translated audio to text
  * Stored the text in a text file


2. File Parsing and Storage
- Parser:

  * Breaks the text file into chunks, extracting relevant information.
  * Organizes content into sections for effective storage.

3. LangChain Model
- OpenAI Integration:

  * Connects to OpenAI Embedding resource for natural language processing.
- Langchain Summary Chain

  * employs map-reduce techniques, ensuring that each chunk of information was succinctly summarized to provide relevant outputs.


- Context Passing:

  * LangChain passes context to OPENAI LLM, ensuring accurate responses.

 
## Features
1. Efficient Summary
- Users can summarise any PDF document
2. mp3 to text conversion
- Users and transcribe any mp3 file



## Future Enhancements
- PDF Search Capabilities: Users will be able to ask questions about the PDF and get accurate answers

## Conclusion
- Our custom GPT model is a one stop solution for summarising any document and creating automatic notes of any lecture file.
