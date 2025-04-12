import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from openrouter import ChatCompletion

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Safety check
if not api_key:
    st.error("API key not found. Please add it to your .env file.")
    st.stop()

# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define LLM using OpenRouter API
class OpenRouterLLM:
    def __init__(self, model="mistralai/mistral-7b-instruct"):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://utd-jsom-chatbot.streamlit.app",
        }

    def __call__(self, prompt, **kwargs):
        response = ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            headers=self.headers,
        )
        return response["choices"][0]["message"]["content"]

llm = OpenRouterLLM()

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
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
