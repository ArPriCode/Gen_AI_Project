import streamlit as st
import numpy as np
import joblib
import re
import textstat

st.set_page_config(
    page_title="Intelligent Exam Question Analysis",
    page_icon="📘",
    layout="wide"
)

lr_model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #f5f7fa;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("System Overview")
    st.markdown("""
    This system analyzes exam questions using:
    - Textual complexity (TF-IDF)
    - Structural features
    - Student response statistics
    
    It predicts the overall difficulty level
    using a trained machine learning model.
    """)

st.markdown("<div class='main-title'>Intelligent Exam Question Analysis System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Driven Difficulty Classification</div>", unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([2,1])

with left:
    question_text = st.text_area("Enter Question Text", height=220)

with right:
    total_students = st.number_input("Total Students Attempted", min_value=1, step=1)
    correct_attempts = st.number_input("Correct Attempts", min_value=0, step=1)
    incorrect_attempts = st.number_input("Incorrect Attempts", min_value=0, step=1)

st.markdown("")

analyze = st.button("Analyze Question", use_container_width=True)

if analyze:

    if question_text.strip() == "":
        st.warning("Please enter question text.")
    elif correct_attempts + incorrect_attempts != total_students:
        st.error("Correct and Incorrect attempts must equal Total Students Attempted.")
    else:

        word_count = len(question_text.split())
        sentence_count = len([s for s in re.split(r'[.!?]+', question_text) if s.strip() != ""])
        readability_score = textstat.flesch_reading_ease(question_text)

        clean_q = clean_text(question_text)

        text_features = vectorizer.transform([clean_q])

        numeric = np.array([[readability_score,
                             word_count,
                             sentence_count,
                             total_students,
                             correct_attempts,
                             incorrect_attempts]])

        numeric_scaled = scaler.transform(numeric)

        final_features = np.hstack((text_features.toarray(), numeric_scaled))

        prediction = lr_model.predict(final_features)[0]
        probabilities = lr_model.predict_proba(final_features)[0]

        st.markdown("---")
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if prediction == "easy":
            st.success("Difficulty Level: EASY")
        elif prediction == "medium":
            st.warning("Difficulty Level: MEDIUM")
        else:
            st.error("Difficulty Level: HARD")

        st.markdown("Confidence Distribution")
        for label, prob in zip(lr_model.classes_, probabilities):
            st.write(f"{label.capitalize()}")
            st.progress(float(prob))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")

        student_accuracy = correct_attempts / total_students
        error_rate = incorrect_attempts / total_students

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Readability Score", round(readability_score, 2))
        a2.metric("Word Count", word_count)
        a3.metric("Student Accuracy", f"{round(student_accuracy*100,2)}%")
        a4.metric("Error Rate", f"{round(error_rate*100,2)}%")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Intelligent Assessment Design Project</p>",
    unsafe_allow_html=True
)