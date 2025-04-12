import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openrouter import ChatOpenRouter

# Load .env variables (works locally; on Streamlit Cloud set secrets instead)
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Check if key is found
if not openrouter_api_key:
    st.error("🔑 OpenRouter API key is missing! Please set it in Streamlit Secrets.")
    st.stop()

# Define LLM with API key
llm = ChatOpenRouter(
    model="mistralai/mistral-7b-instruct",  # Choose from https://openrouter.ai/docs#models
    api_key=openrouter_api_key
)

# Embeddings + FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create Retrieval QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
st.set_page_config(page_title="🤖 JSOM Chatbot – Ask Me Anything")
st.title("🤖 JSOM Chatbot – Ask Me Anything")
query = st.text_input("What would you like to know about JSOM?")

if query:
    with st.spinner("Thinking..."):
        try:
            result = qa.invoke(query)
            st.markdown(f"**Answer:** {result['result']}")
            with st.expander("📄 Context Chunks Used"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content.strip()}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
