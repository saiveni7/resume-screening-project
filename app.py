import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening System")

job_description = st.text_area("Enter Job Description")
resume = st.text_area("Paste Resume Text")

if st.button("Check Match"):
    documents = [job_description, resume]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    score = similarity[0][0] * 100
    
    st.write(f"Matching Score: {score:.2f}%")
    
    if score > 70:
        st.success("Strong Match ✅")
    else:
        st.warning("Not a Strong Match ❌")