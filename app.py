import os
import streamlit as st
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Embeddings using OpenRouter
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Load FAISS vectorstore
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM setup (ChatOpenAI via OpenRouter)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct"
)

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
