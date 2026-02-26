import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
import string
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

st.title("AI Resume Screening System")

# Step 1: Job Description Input
job_description = st.text_area("Enter Job Description")

# Step 2: Resume PDF Upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
resume = ""

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        resume += page.extract_text()

# Step 3: Check Match Button
if st.button("Check Match"):
    if job_description and resume:

        # Step 3a: Preprocess Text
        def preprocess(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = text.split()
            words = [word for word in words if word not in stopwords.words('english')]
            return " ".join(words)

        documents = [preprocess(job_description), preprocess(resume)]

        # Step 3b: TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        score = similarity[0][0] * 100

        # Step 3c: Show Matching Score
        st.write(f"Matching Score: {score:.2f}%")
        if score > 70:
            st.success("Strong Match ✅")
        elif score > 40:
            st.warning("Moderate Match ⚠")
        else:
            st.error("Low Match ❌")

        # Step 3d: Graph
        st.subheader("Match Score Graph")
        fig, ax = plt.subplots()
        ax.bar(["Match Score"], [score], color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        st.pyplot(fig)

    else:
        st.warning("Please enter Job Description and upload Resume")