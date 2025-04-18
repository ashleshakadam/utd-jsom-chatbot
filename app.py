import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import requests

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter

# Load latest crawled site content
loader = JSONLoader(file_path="data/full_site.json", jq_schema=".values()", text_content=True)
documents = loader.load()

# Split into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create embeddings & vectorstore dynamically
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define query function
def query_openrouter(prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://utd-jsom-chatbot.streamlit.app"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Streamlit Config
st.set_page_config(page_title="🤖 JSOM Chatbot", layout="wide")

# Styling
st.markdown("""
    <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #0d3c61;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .answer-box {
            background-color: #f0f6ff;
            border-left: 5px solid #0078d4;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
            font-size: 1.05rem;
        }
        .chunk-box {
            background-color: #f9f9f9;
            border: 1px dashed #ccc;
            border-radius: 5px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .footer {
            text-align: center;
            color: #777;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🤖 JSOM Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart answers from the Jindal School of Management website</div>', unsafe_allow_html=True)

# Input box
query = st.text_input("🔍 What would you like to know about JSOM?", "")

if query:
    with st.spinner("🤔 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"Use the following JSOM information to answer:\n\n{context}\n\nQuestion: {query}"
            answer = query_openrouter(prompt, OPENROUTER_API_KEY)

            st.markdown(f'<div class="answer-box"><strong>Answer:</strong> {answer}</div>', unsafe_allow_html=True)

            with st.expander("📄 Context Snippets Used"):
                for i, doc in enumerate(docs):
                    st.markdown(f'<div class="chunk-box"><strong>Chunk {i+1}:</strong> {doc.page_content.strip()}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
last_updated = datetime.now().strftime("%B %d, %Y at %I:%M %p")
st.markdown(f'<div class="footer">📅 Last updated: {last_updated}</div>', unsafe_allow_html=True)
