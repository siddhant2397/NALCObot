import streamlit as st
import requests
import os
import pandas as pd
from PIL import Image
from docx import Document
import fitz  # PyMuPDF
import openai
import numpy as np
import uuid
import psycopg2
from psycopg2.extras import Json
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Load secrets from .streamlit/secrets.toml
openai.api_key = st.secrets["openai_api_key"]
OCR_SPACE_API_KEY = st.secrets["ocr_space_api_key"]

SUPABASE_DB_URL = st.secrets["supabase_db_url"]
SUPABASE_DB_USER = st.secrets["supabase_db_user"]
SUPABASE_DB_PASS = st.secrets["supabase_db_pass"]
SUPABASE_DB_NAME = st.secrets["supabase_db_name"]

GITHUB_USER = "siddhant2397"
GITHUB_REPO = "NALCObot"
GITHUB_BRANCH = "main"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

@st.cache_data(ttl=3600)
def list_github_files(user, repo, branch="main"):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/"
    response = requests.get(api_url)
    files = response.json()
    return [f for f in files if f['name'].endswith((".docx", ".xlsx", ".pdf"))]

def fetch_file(file):
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{file['name']}"
    response = requests.get(raw_url)
    with open(file['name'], "wb") as f:
        f.write(response.content)
    return file['name']

def ocr_space_image(filepath):
    with open(filepath, 'rb') as f:
        response = requests.post(
            url='https://api.ocr.space/parse/image',
            files={'file': f},
            data={'apikey': OCR_SPACE_API_KEY}
        )
    result = response.json()
    return result['ParsedResults'][0]['ParsedText'] if 'ParsedResults' in result else ""

def extract_text(filepath):
    if filepath.endswith(".docx"):
        return extract_docx_text(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
        return df.to_string(index=False)
    elif filepath.endswith(".pdf"):
        return extract_pdf_text(filepath)
    return ""

def extract_docx_text(filepath):
    doc = Document(filepath)
    text = "\n".join([p.text for p in doc.paragraphs])
    for rel in doc.part._rels:
        rel = doc.part._rels[rel]
        if "image" in rel.target_ref:
            img_data = rel.target_part.blob
            with open("temp.png", "wb") as f:
                f.write(img_data)
            text += "\n" + ocr_space_image("temp.png")
            os.remove("temp.png")
    return text

def extract_pdf_text(filepath):
    doc = fitz.open(filepath)
    text = " ".join([page.get_text() for page in doc])
    if len(text.strip()) < 100:
        images = [page.get_pixmap().pil_tobytes(format="PNG") for page in doc]
        for i, img_bytes in enumerate(images):
            with open(f"page_{i}.png", "wb") as f:
                f.write(img_bytes)
        text = ""
        for i in range(len(images)):
            text += ocr_space_image(f"page_{i}.png") + "\n"
            os.remove(f"page_{i}.png")
    return text

def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def connect_db():
    return psycopg2.connect(
        host=SUPABASE_DB_URL,
        dbname=SUPABASE_DB_NAME,
        user=SUPABASE_DB_USER,
        password=SUPABASE_DB_PASS,
        sslmode="require"
    )

def store_embeddings(chunks):
    conn = connect_db()
    cur = conn.cursor()
    for chunk in chunks:
        emb = embed_text(chunk).tolist()
        cur.execute(
            "INSERT INTO documents (id, content, embedding) VALUES (%s, %s, %s)",
            (str(uuid.uuid4()), chunk, Json(emb))
        )
    conn.commit()
    cur.close()
    conn.close()

def log_interaction(question, input_tokens, output_tokens, cost):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id UUID PRIMARY KEY,
            timestamp TIMESTAMP,
            question TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            cost NUMERIC
        )
    """)
    cur.execute("""
        INSERT INTO chat_logs (id, timestamp, question, prompt_tokens, completion_tokens, cost)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), datetime.utcnow(), question, input_tokens, output_tokens, cost))
    conn.commit()
    cur.close()
    conn.close()

def get_monthly_usage():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(SUM(prompt_tokens), 0),
               COALESCE(SUM(completion_tokens), 0),
               COALESCE(SUM(cost), 0)
        FROM chat_logs
        WHERE DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', CURRENT_DATE)
    """)
    result = cur.fetchone()
    conn.close()
    return result

def get_top_chunks(question_embedding, k=3):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT content, embedding FROM documents")
    results = cur.fetchall()
    conn.close()

    chunks = [r[0] for r in results]
    embeddings = [np.array(r[1], dtype=np.float32) for r in results]
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_idx]

def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ask_gpt(question, context):
    prompt = f"Answer the following question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400
    )
    answer = response['choices'][0]['message']['content']
    usage = response['usage']
    input_tokens = usage['prompt_tokens']
    output_tokens = usage['completion_tokens']
    cost = (input_tokens / 1000 * 0.0005) + (output_tokens / 1000 * 0.0015)
    return answer, input_tokens, output_tokens, cost

st.title("📄 Organizational Document Chatbot (GPT-3.5 + OCR + Usage Tracker)")

question = st.text_input("Ask your question about the documents")

if question:
    st.info("📂 Fetching and processing documents from GitHub...")
    files = list_github_files(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH)
    all_chunks = []

    for file in files:
        local_name = fetch_file(file)
        text = extract_text(local_name)
        chunks = split_text(text)
        all_chunks.extend(chunks)
        os.remove(local_name)

    st.success(f"✅ Loaded {len(files)} files with {len(all_chunks)} chunks.")
    store_embeddings(all_chunks)

    question_embedding = embed_text(question)
    top_chunks = get_top_chunks(question_embedding, k=3)
    context = "\n---\n".join(top_chunks)

    st.info("💬 Generating answer from GPT-3.5...")
    answer, input_tokens, output_tokens, cost = ask_gpt(question, context)
    log_interaction(question, input_tokens, output_tokens, cost)

    st.success("✅ Answer:")
    st.write(answer)

    st.markdown("---")
    st.subheader("📊 Cost Summary for This Query")
    st.write(f"**Prompt tokens:** {input_tokens}")
    st.write(f"**Response tokens:** {output_tokens}")
    st.write(f"**Estimated cost:** ${cost:.6f} USD")

    prompt_sum, completion_sum, cost_sum = get_monthly_usage()
    st.markdown("---")
    st.subheader("📆 Monthly Usage Summary")
    st.write(f"**Total input tokens:** {prompt_sum}")
    st.write(f"**Total output tokens:** {completion_sum}")
    st.write(f"**Estimated monthly cost:** ${cost_sum:.4f} USD")
