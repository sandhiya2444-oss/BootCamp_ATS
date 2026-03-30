%%writefile app.py
import streamlit as st
import PyPDF2
import re
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

# Load AI Model (Cached)
@st.cache_resource
def load_ai():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI - Upload Resume & Job Description
st.title("🤖 FREE AI ATS Scanner (2026 Edition)")
st.markdown("**Keyword + AI Context = Resume Shortlist**")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Resume PDF", type="pdf")
with col2:
    jd_text = st.text_area("📋 Job Description", height=200,
                          placeholder="Python developer with Git, data analysis...")

# Run ATS Analysis
if resume_file and jd_text and st.button("🚀 AI SCAN", use_container_width=True):
    model = load_ai()

    with st.spinner("Analyzing resume..."):
        # --- Keyword Score ---
        pdf = PyPDF2.PdfReader(resume_file)
        resume_text = " ".join(p.extract_text().lower() for p in pdf.pages)

        # Extract words
        jd_keywords = re.findall(r'\b[a-z]{3,12}\b', jd_text.lower())
        jd_keywords = [w for w in set(jd_keywords) if w.isalpha()][:12]  # Top 12 keywords
        resume_words = re.findall(r'\b[a-z]{3,12}\b', resume_text)
        counts = Counter(resume_words)

        keyword_score = round(sum(counts.get(kw, 0) for kw in jd_keywords) / len(jd_keywords) * 100)

        # --- Contextual AI Score ---
        resume_sentences = re.split(r'[.!?]+', resume_text)
        jd_sentences = re.split(r'[.!?]+', jd_text)

        resume_emb = model.encode(resume_sentences)
        jd_emb = model.encode(jd_sentences)

        similarities = []
        for r_emb in resume_emb:
            for j_emb in jd_emb:
                sim = np.dot(r_emb, j_emb) / (np.linalg.norm(r_emb) * np.linalg.norm(j_emb))
                similarities.append(sim)

        ai_score = round(np.mean([s for s in similarities if s > 0.3]) * 100)

        # --- Strengths & Weaknesses ---
        strengths = [kw.upper() for kw in jd_keywords if counts.get(kw, 0) > 1]
        weaknesses = [kw.upper() for kw in jd_keywords if counts.get(kw, 0) == 0][:3]

    # Display Results
    st.markdown("## 🎯 Dual Engine Scores")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 Keyword Match", f"{keyword_score}%")
        if strengths:
            st.success(f"✅ Strengths: {', '.join(strengths[:3])}")

    with col2:
        st.metric("🧠 AI Context", f"{ai_score}%")
        if weaknesses:
            st.error(f"❌ Fix: Add {', '.join(weaknesses)}")

    st.info("💡 Pro Tip: Rewrite bullets using LLMs to improve score")
    st.balloons()

#Seperate Cell

from pyngrok import ngrok
import subprocess
import os

# Kill any running ngrok tunnels
ngrok.kill()

# Start a streamlit process in the background
# It will run on port 8501 by default
cmd = ["streamlit", "run", "app.py", "--server.port", "8501"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Get a public URL for the streamlit app
public_url = ngrok.connect(8501)

print("Your Streamlit app is running at:", public_url)


#Requirements

!pip install streamlit
!pip install PyPDF2
!pip install re
!pip install numpy
!pip install Counter
!pip install pyngrok










