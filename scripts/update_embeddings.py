# File: scripts/update_embeddings.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from scripts.utils.io import load_json
from scripts.utils.logging import log_info

# Load changes
changes = load_json("data/changes.json")
if not changes["added"] and not changes["updated"]:
    print("🔄 No updated pages. Skipping vectorstore rebuild.")
    exit(0)

# Load pages to embed
data = load_json("data/full_site.json")
docs = []

for url in changes["added"] + changes["updated"]:
    text = data.get(url, "")
    if text:
        docs.append(f"{url}\n\n{text}")

# Split and embed
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.create_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Save
vectorstore.save_local("embeddings/faiss_index")
log_info("Vectorstore updated with changed content.")
print("✅ Vectorstore updated.")
