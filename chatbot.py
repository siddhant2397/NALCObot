import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ---------------- Setup ---------------- #
st.set_page_config(page_title="CISF Chatbot", layout="wide")
st.title("📚 Chatbot")

# ---------------- Load Files ---------------- #
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

documents = []
for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)
    try:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    except Exception as e:
        st.error(f"❌ Error loading {file}: {e}")

if documents:
    st.success(f"✅ {len(documents)} document(s) loaded successfully.")
else:
    st.warning("⚠️ No documents found in the 'data/' folder.")

# ---------------- Embeddings & Vector Store ---------------- #
if documents:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

# ---------------- Query Section ---------------- #
if documents:
    query = st.text_input("💬 Ask a question:")

    if query:
        if len(query) > 100:
            st.warning("❗ Query too long. Limit it to 100 characters.")
            st.stop()

        with st.spinner("🔍 Searching..."):
            docs = retriever.get_relevant_documents(query)
            if docs:
                top = docs[0].page_content
                st.success("✅ Answer:")
                st.write(top[:500] + "..." if len(top) > 500 else top)
            else:
                st.warning("No relevant content found.")
