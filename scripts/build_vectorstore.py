from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

print("🔑 OpenAI key starts with:", os.getenv("OPENAI_API_KEY")[:10])

# Load file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, "data", "admissions.txt")

with open(input_path, "r") as f:
    raw_text = f.read()

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents([Document(page_content=raw_text)])

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(os.path.join(project_root, "embeddings", "faiss_index"))
