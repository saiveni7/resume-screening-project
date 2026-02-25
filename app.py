import streamlit as st
import nltk
nltk.download('stopwords')  # Fix: download stopwords for Streamlit Cloud
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to clean text
def preprocess(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Streamlit UI
st.title("AI Resume Screening System")

# Job Description Input
job_desc = st.text_area("Enter Job Description")

# Resume Upload
resume_file = st.file_uploader("Upload Resume (Text file only)", type=["txt"])

# Check Match Button
if st.button("Check Match"):
    if job_desc and resume_file is not None:
        # Read resume content
        resume_text = resume_file.read().decode("utf-8")
        
        # Preprocess texts
        documents = [preprocess(job_desc), preprocess(resume_text)]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = similarity[0][0] * 100
        
        # Show result
        st.subheader(f"Matching Score: {score:.2f}%")
        if score > 70:
            st.success("Strong Match ✅")
        elif score > 40:
            st.warning("Moderate Match ⚠")
        else:
            st.error("Low Match ❌")
    else:
        st.warning("Please enter Job Description and upload Resume.")