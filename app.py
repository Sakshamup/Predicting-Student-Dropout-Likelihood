import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("trained_dropout_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# --- Load Encoders ---
@st.cache_resource
def load_encoders():
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return encoders

# --- Load Feature Order ---
@st.cache_resource
def load_feature_names():
    with open("feature_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
encoders = load_encoders()
feature_order = load_feature_names()

# --- App UI ---
st.title("🎓 Student Dropout Prediction App")

st.sidebar.header("📋 Enter Student Details")

# Sidebar Inputs
age = st.sidebar.slider("Age", 15, 50, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
course = st.sidebar.selectbox("Course Name", ["Python", "Machine Learning", "Data Science", "AI", "Others"])
learning_style = st.sidebar.selectbox("Learning Style", ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
engagement = st.sidebar.selectbox("Engagement Level", ["Low", "Medium", "High"])
time_videos = st.sidebar.slider("Time Spent on Videos (hrs)", 0.0, 100.0, 20.0)
quiz_attempts = st.sidebar.slider("Quiz Attempts", 0, 20, 3)
quiz_scores = st.sidebar.slider("Quiz Scores", 0, 100, 70)
forum = st.sidebar.slider("Forum Participation", 0, 100, 10)
assignment_completion = st.sidebar.slider("Assignment Completion Rate (%)", 0, 100, 80)
final_exam = st.sidebar.slider("Final Exam Score", 0, 100, 75)
feedback = st.sidebar.slider("Feedback Score", 0, 10, 8)

# Build input DataFrame
input_dict = {
    "Age": age,
    "Gender": gender,
    "Education_Level": education,
    "Course_Name": course,
    "Learning_Style": learning_style,
    "Engagement_Level": engagement,
    "Time_Spent_on_Videos": time_videos,
    "Quiz_Attempts": quiz_attempts,
    "Quiz_Scores": quiz_scores,
    "Forum_Participation": forum,
    "Assignment_Completion_Rate": assignment_completion,
    "Final_Exam_Score": final_exam,
    "Feedback_Score": feedback,
}

input_df = pd.DataFrame([input_dict])

# Encode categorical columns safely
for col in input_df.select_dtypes(include='object').columns:
    le = encoders[col]
    input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

# Reorder columns to match training
input_df = input_df[feature_order]

# Predict
if st.button("🔍 Predict Dropout Likelihood"):
    prediction = model.predict(input_df)[0]
    prediction_label = encoders["Dropout_Likelihood"].inverse_transform([prediction])[0]
    st.success(f" Predicted Dropout Likelihood: **{prediction_label}**")

    # Show prediction probabilities
    proba = model.predict_proba(input_df)[0]
    st.write("Prediction Confidence:")
    st.write({encoders["Dropout_Likelihood"].inverse_transform([i])[0]: f"{p:.2%}" for i, p in enumerate(proba)})

# Optional: Feature importances
if hasattr(model, "feature_importances_"):
    st.subheader(" Feature Importances")
    feat_importances = pd.Series(model.feature_importances_, index=feature_order)
    st.bar_chart(feat_importances.sort_values(ascending=False))
