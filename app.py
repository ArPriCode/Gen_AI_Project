import streamlit as st
import numpy as np
import joblib
import re

# 1. Load the pre-trained artifacts efficiently
@st.cache_resource
def load_models():
    dt_model = joblib.load("dt_model.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return dt_model, vectorizer, scaler, encoder

try:
    dt_model, vectorizer, scaler, encoder = load_models()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- UI Setup ---
st.title("Exam Question Difficulty Predictor")
st.write("Enter the question details below to predict difficulty.")

# Question Text Input
question_text = st.text_area("Question Text", placeholder="Type the exam question here...")

# Cognitive Level Buttons (Horizontal Radio)
st.subheader("Cognitive Level")
bloom_options = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
cognitive_level = st.radio("Select Bloom's Taxonomy Level:", bloom_options, horizontal=True)

# Student Response Metrics
st.subheader("Student Response Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    total_students_attempted = st.number_input("Total Attempts", min_value=0, value=0)
with col2:
    correct_attempts = st.number_input("Correct Attempts", min_value=0, value=0)
with col3:
    incorrect_attempts = st.number_input("Incorrect Attempts", min_value=0, value=0)

# Predict Button Logic
if st.button("Predict Difficulty", type="primary"):
    
    # --- VALIDATION CHECKS ---
    if not question_text.strip():
        st.warning("Please enter the question text before predicting.")
    elif total_students_attempted == 0:
         st.warning("Total attempts must be greater than 0 to make a prediction.")
    elif (correct_attempts + incorrect_attempts) != total_students_attempted:
        # Fails the prediction if the math doesn't add up
        st.error(f"Error: Correct ({correct_attempts}) + Incorrect ({incorrect_attempts}) must equal Total Attempts ({total_students_attempted}).")
    else:
        # --- PREDICTION LOGIC ---
        try:
            # Clean and Transform text
            cleaned_text = clean_text(question_text)
            text_vectorized = vectorizer.transform([cleaned_text]).toarray()
            
            # Prepare and scale numeric features
            numeric_inputs = [[total_students_attempted, correct_attempts, incorrect_attempts]]
            numeric_scaled = scaler.transform(numeric_inputs)
            
            # Encode categorical feature (lowercase to match training data)
            cat_input = [[cognitive_level.lower()]]
            cat_encoded = encoder.transform(cat_input)
            
            # Combine all features
            final_features = np.hstack((text_vectorized, numeric_scaled, cat_encoded))
            
            # Predict
            prediction = dt_model.predict(final_features)[0]
            
            st.success(f"Predicted Difficulty: **{prediction.upper()}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")