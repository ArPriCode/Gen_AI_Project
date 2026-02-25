import streamlit as st
import numpy as np
import re
import joblib
from sentence_transformers import SentenceTransformer


lr_model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title="Exam Difficulty Analyzer")

st.title("📘 Exam Question Difficulty Analyzer")

st.write("Predict difficulty using question text and student response data.")

question_text = st.text_area("Enter Question Text")

readability_score = st.number_input("Readability Score", min_value=0.0)
word_count = st.number_input("Word Count", min_value=0)
sentence_count = st.number_input("Sentence Count", min_value=0)

total_students = st.number_input("Total Students Attempted", min_value=1)
correct_attempts = st.number_input("Correct Attempts", min_value=0)

if st.button("Predict Difficulty"):

    if question_text.strip() == "":
        st.warning("Please enter question text.")
    else:

        clean_q = clean_text(question_text)
        emb = embedder.encode([clean_q])
        student_accuracy = correct_attempts / total_students
        error_rate = 1 - student_accuracy

        numeric = np.array([[readability_score,
                             word_count,
                             sentence_count,
                             student_accuracy,
                             error_rate]])

        numeric_scaled = scaler.transform(numeric)
        features = np.hstack((emb, numeric_scaled))

        prediction = lr_model.predict(features)[0]
        probabilities = lr_model.predict_proba(features)[0]


        st.success(f"Predicted Difficulty: {prediction.upper()}")

        st.subheader("Prediction Confidence")

        class_labels = lr_model.classes_

        for i, label in enumerate(class_labels):
            st.write(f"{label}: {round(probabilities[i]*100, 2)}%")

        st.subheader("Student Performance Analysis")

        st.write(f"Student Accuracy: {round(student_accuracy*100,2)}%")
        st.write(f"Error Rate: {round(error_rate*100,2)}%")