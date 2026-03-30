# UTD JSOM Chatbot

## Overview
A retrieval-augmented chatbot built to answer questions over university and school-related content using semantic search and local language model inference.

The project demonstrates an end-to-end GenAI pipeline that combines document ingestion, embedding generation, vector retrieval, and answer synthesis in an interactive application.

## Business Problem
Students and stakeholders often need quick answers from large collections of institutional documents, but relevant information is typically distributed across PDFs, webpages, and long-form text.

Traditional keyword search struggles with paraphrased questions, multi-step retrieval, and context preservation.

## Solution
This project implements a RAG system that:
- ingests document collections
- chunks and embeds text
- stores embeddings in FAISS
- retrieves semantically relevant context
- generates grounded responses using a local language model

## Architecture
The application is built around a standard retrieval pipeline:

- document ingestion and preprocessing
- chunking and embedding generation
- vector storage with FAISS
- query embedding and similarity retrieval
- answer generation using Mistral through Ollama
- Streamlit interface for user interaction

## Methodology

### Retrieval Layer
Documents are embedded and stored in a vector index to support semantic similarity search.

### Prompt Construction
Relevant chunks are injected into the response prompt to ground model outputs in retrieved evidence.

### Generation Layer
A local LLM is used to generate answers over retrieved context rather than answering from parametric memory alone.

### User Interface
A Streamlit application provides a simple front end for question submission and answer delivery.

## Results
The system demonstrates:
- improved answer relevance over naive keyword search
- practical retrieval-augmented response generation over institutional content
- a deployable local GenAI workflow using open-source tools

## Tech Stack
Python, LangChain, FAISS, Streamlit, Ollama, Mistral

## Repository Structure
```text
utd-jsom-chatbot/
├── app/
├── data/
├── embeddings/
├── vectorstore/
├── utils/
└── README.md
```

## How to Run
```text
git clone https://github.com/ashleshakadam/utd-jsom-chatbot.git
cd utd-jsom-chatbot
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements
	•	add source citation display in responses
	•	evaluate retrieval quality with benchmark queries
	•	introduce reranking for improved context selection
	•	support hybrid retrieval over structured and unstructured sources

## Author
Ashlesha Kadam
