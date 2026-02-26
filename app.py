import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.title("AI Resume Screening System")

# Job Description input
job_description = st.text_area("Enter Job Description")

# Resume PDF upload and text extraction
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
resume = ""
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            resume += text
    # Debug print (optional, comment out if not needed)
    st.write("Extracted Resume Text Preview:")
    st.write(resume[:300] + "..." if len(resume) > 300 else resume)

# Button to check match
if st.button("Check Match"):
    if job_description.strip() == "" or resume.strip() == "":
        st.warning("Please enter Job Description and upload Resume")
    else:
        documents = [job_description, resume]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        score = similarity[0][0] * 100

        st.write(f"Matching Score: {score:.2f}%")

        # Display status message
        if score > 70:
            st.success("Strong Match ✅")
        elif score > 40:
            st.info("Average Match ⚠️")
        else:
            st.error("Low Match ❌")

        # Display bar graph of score
        st.subheader("Match Score Graph")
        fig, ax = plt.subplots()
        ax.bar(["Match Score"], [score], color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        st.pyplot(fig)