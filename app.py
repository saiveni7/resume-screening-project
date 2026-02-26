import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.title("AI Resume Screening System")

# Job Description Input
job_description = st.text_area("Enter Job Description")

# PDF Resume Upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

resume = ""
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            resume += text

    if resume:
        st.subheader("Resume Text Preview (first 500 characters):")
        st.write(resume[:500])
    else:
        st.warning("⚠️ Could not extract text. Please upload a text-based PDF, not a scanned image.")

# Matching Button
if st.button("Check Match"):
    if not job_description or not resume:
        st.warning("Please enter Job Description and upload Resume")
    else:
        documents = [job_description, resume]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        score = similarity[0][0] * 100

        st.write(f"Matching Score: {score:.2f}%")

        if score > 70:
            st.success("Strong Match ✅")
        elif score > 40:
            st.info("Average Match 🙂")
        else:
            st.warning("Low Match ❌")

        # Graph
        st.subheader("Match Score Graph")
        fig, ax = plt.subplots()
        ax.bar(["Match Score"], [score])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        st.pyplot(fig)