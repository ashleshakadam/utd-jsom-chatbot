import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import List
import requests


# Load .env variables locally
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Define minimal OpenRouterLLM wrapper
class OpenRouterLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "mistral",  # or any OpenRouter-supported model
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self):
        return "custom_openrouter"

# Build vectorstore at runtime
loader = TextLoader("data/admissions.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Setup LLM and RetrievalQA
llm = OpenRouterLLM()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# UI
st.set_page_config(page_title="🤖 JSOM Chatbot")
st.title("🤖 JSOM Chatbot – Ask Me Anything")
query = st.text_input("What would you like to know about JSOM?")

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke(query)
        st.markdown(f"**Answer:** {result['result']}")
        with st.expander("📄 Context Chunks Used"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:** {doc.page_content.strip()}")
