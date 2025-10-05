# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:00:19 2025

@author: sksou
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import openai


# ------------------ SETUP ------------------
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Load saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Configure OpenAI API key (store securely in Streamlit Cloud or local .streamlit/secrets.toml)
openai.api_key = st.secrets["OPENAI_API_KEY"]


# ------------------ SIDEBAR ------------------
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )


# ------------------ DIABETES PREDICTION ------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    user_input = []

    if st.button('Diabetes Test Result'):
        try:
            user_input = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), int(Age)
            ]
            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = "Please enter valid numeric values."

    st.success(diab_diagnosis)

    # --- AI Chatbot ---
    if diab_diagnosis:
        st.subheader("💬 AI Health Assistant")
        user_query = st.text_input("Ask about your diabetic condition 👇")

        if user_query:
            with st.spinner("HealthBot is thinking..."):
                disease_context = f"User inputs: {user_input}. Diagnosis: {diab_diagnosis}."
                prompt = (
                    f"You are a helpful and safe medical advice assistant. "
                    f"Based on {disease_context}, give helpful, non-prescriptive advice "
                    f"for the user's question: {user_query}. "
                    f"Focus only on diabetes-related tips, lifestyle, diet, and precautions. "
                    f"Do not provide medical diagnosis or prescriptions."
                )

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.7
                )

                st.write("🤖 HealthBot:", response['choices'][0]['message']['content'])


# ------------------ HEART DISEASE PREDICTION ------------------
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''
    user_input = []

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [
                int(age), int(sex), int(cp), int(trestbps), int(chol),
                int(fbs), int(restecg), int(thalach), int(exang),
                float(oldpeak), int(slope), int(ca), int(thal)
            ]
            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
        except ValueError:
            heart_diagnosis = "Please enter valid numeric values."

    st.success(heart_diagnosis)

    # --- AI Chatbot ---
    if heart_diagnosis:
        st.subheader("💬 AI Health Assistant")
        user_query = st.text_input("Ask about your heart health 👇")

        if user_query:
            with st.spinner("HealthBot is analyzing..."):
                disease_context = f"User inputs: {user_input}. Diagnosis: {heart_diagnosis}."
                prompt = (
                    f"You are a safe and informative health assistant. Based on {disease_context}, "
                    f"provide helpful, general advice to the user about heart disease prevention, "
                    f"lifestyle, diet, or exercise. Avoid medical diagnosis or prescribing medicines. "
                    f"User question: {user_query}"
                )

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.7
                )

                st.write("🤖 HealthBot:", response['choices'][0]['message']['content'])


# ------------------ PARKINSON'S PREDICTION ------------------
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    fields = [
        "fo", "fhi", "flo", "Jitter_percent", "Jitter_Abs", "RAP", "PPQ", "DDP",
        "Shimmer", "Shimmer_dB", "APQ3", "APQ5", "APQ", "DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    inputs = []
    for i, field in enumerate(fields):
        with [col1, col2, col3, col4, col5][i % 5]:
            val = st.text_input(field)
            inputs.append(val)

    parkinsons_diagnosis = ''
    user_input = []

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(x) for x in inputs]
            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = "⚠️ Please enter valid numeric values."

    st.success(parkinsons_diagnosis)

    # --- AI Chatbot ---
    if parkinsons_diagnosis:
        st.subheader("💬 AI Health Assistant")
        user_query = st.text_input("Ask about Parkinson’s disease 👇")

        if user_query:
            with st.spinner("HealthBot is generating advice..."):
                disease_context = f"User inputs: {user_input}. Diagnosis: {parkinsons_diagnosis}."
                prompt = (
                    f"You are a helpful assistant. Based on {disease_context}, provide helpful and "
                    f"safe suggestions related to Parkinson’s disease — focusing on exercise, therapy, "
                    f"and lifestyle management. Avoid diagnosis or medication suggestions. "
                    f"User question: {user_query}"
                )

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.7
                )

                st.write("🤖 HealthBot:", response['choices'][0]['message']['content'])
