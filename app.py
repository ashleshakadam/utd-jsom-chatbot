import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openrouter import OpenRouterLLM  # ✅ Correct for version 0.0.1

# Load .env variables
load_dotenv()

# Initialize OpenRouter LLM
llm = OpenRouterLLM(api_key=os.getenv("OPENROUTER_API_KEY"))

# Load FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# UI
st.set_page_config(page_title="🤖 JSOM Chatbot – Ask Me Anything")
st.title("🤖 JSOM Chatbot – Ask Me Anything")
query = st.text_input("What would you like to know about JSOM?")

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke(query)
        st.markdown(f"**Answer:** {result['result']}")

        with st.expander("📄 Context Chunks Used"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:** {doc.page_content.strip()}")
