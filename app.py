import os
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment variable
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load and split documents
loader = TextLoader("data/admissions.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Build embeddings and FAISS vectorstore at runtime
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# OpenRouter query function
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

# Streamlit UI
st.set_page_config(page_title="🤖 JSOM Chatbot – Ask Me Anything")
st.title("🤖 JSOM Chatbot – Ask Me Anything")

query = st.text_input("What would you like to know about JSOM?")

if query:
    with st.spinner("Searching JSOM knowledge base..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"Use the following JSOM information to answer:\n\n{context}\n\nQuestion: {query}"
            answer = query_openrouter(prompt, OPENROUTER_API_KEY)

            st.markdown(f"**Answer:** {answer}")

            with st.expander("📄 Source Chunks"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content.strip()}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
