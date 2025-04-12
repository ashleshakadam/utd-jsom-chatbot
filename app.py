import os
import streamlit as st
from dotenv import load_dotenv
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load FAISS vectorstore from local directory
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Query OpenRouter API directly via HTTPS
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
    with st.spinner("Searching knowledge base..."):
        try:
            # Retrieve relevant chunks
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Append context to the prompt
            full_prompt = f"Use the following JSOM information to answer:\n\n{context}\n\nQuestion: {query}"
            answer = query_openrouter(full_prompt, OPENROUTER_API_KEY)

            # Display answer
            st.markdown(f"**Answer:** {answer}")

            # Display source chunks
            with st.expander("📄 Context Chunks Used"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content.strip()}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
