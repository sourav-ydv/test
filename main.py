# -*- coding: utf-8 -*-
"""
Multiple Disease Prediction + Chatbot
Created on Mon Sep 29 2025
@author: sksou
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from transformers import pipeline

# =========================
# Hugging Face Chatbot Setup
# =========================
@st.cache_resource
def load_chatbot():
    return pipeline("text2text-generation", model="google/flan-t5-base")

chatbot = load_chatbot()

# =========================
# Helper: AI Chatbot
# =========================
def get_chatbot_response(disease, diagnosis, user_query="", summary=False):
    if summary:
        prompt = f"Summarize the condition and advice for a patient diagnosed with {disease}: {diagnosis}"
    else:
        prompt = f"""
        You are a medical assistant chatbot.
        The patient is diagnosed with {disease}: {diagnosis}.
        The user is asking: {user_query}
        Only answer questions related to {disease}. 
        If the question is unrelated, politely refuse.
        Provide suggestions, precautions, severity, lifestyle advice, and include a disclaimer.
        """
    output = chatbot(prompt, max_length=256, do_sample=True)
    return output[0]['generated_text']

# =========================
# Load Saved Models
# =========================
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        icons=["activity", "heart", "person"],
        default_index=0
    )

# =========================
# Diabetes Prediction Page
# =========================
if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input("Number of Pregnancies")
    with col2: Glucose = st.text_input("Glucose Level")
    with col3: BloodPressure = st.text_input("Blood Pressure")
    with col1: SkinThickness = st.text_input("Skin Thickness")
    with col2: Insulin = st.text_input("Insulin Level")
    with col3: BMI = st.text_input("BMI")
    with col1: DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2: Age = st.text_input("Age")

    if "diab_diagnosis" not in st.session_state:
        st.session_state.diab_diagnosis = ""
    if "chat_diab" not in st.session_state:
        st.session_state.chat_diab = []

    if st.button("🔍 Get Diabetes Test Result"):
        try:
            user_input = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), int(Age)
            ]
            pred = diabetes_model.predict([user_input])
            st.session_state.diab_diagnosis = "Diabetic" if pred[0] == 1 else "Not Diabetic"
        except:
            st.session_state.diab_diagnosis = "⚠️ Invalid input"

    if st.session_state.diab_diagnosis:
        st.success(f"Diagnosis: {st.session_state.diab_diagnosis}")

        st.subheader("💬 Diabetes Assistant Chatbot")
        for msg in st.session_state.chat_diab:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("📋 Summary of Condition"):
            summary_text = get_chatbot_response("Diabetes", st.session_state.diab_diagnosis, summary=True)
            st.info(summary_text)

        if user_query := st.chat_input("Ask about your diabetes condition..."):
            st.session_state.chat_diab.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Diabetes", st.session_state.diab_diagnosis, user_query)
            st.session_state.chat_diab.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)

# =========================
# Heart Disease Prediction Page
# =========================
if selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input("Age")
    with col2: sex = st.text_input("Sex (0=F,1=M)")
    with col3: cp = st.text_input("Chest Pain Type")
    with col1: trestbps = st.text_input("Resting BP")
    with col2: chol = st.text_input("Cholesterol")
    with col3: fbs = st.text_input("Fasting Blood Sugar >120 mg/dl (0/1)")
    with col1: restecg = st.text_input("Resting ECG")
    with col2: thalach = st.text_input("Max Heart Rate")
    with col3: exang = st.text_input("Exercise Angina (0/1)")
    with col1: oldpeak = st.text_input("ST Depression")
    with col2: slope = st.text_input("Slope of ST")
    with col3: ca = st.text_input("Major vessels colored")
    with col1: thal = st.text_input("Thal (0-3)")

    if "heart_diagnosis" not in st.session_state:
        st.session_state.heart_diagnosis = ""
    if "chat_heart" not in st.session_state:
        st.session_state.chat_heart = []

    if st.button("🔍 Get Heart Disease Test Result"):
        try:
            user_input = [
                int(age), int(sex), int(cp), int(trestbps), int(chol),
                int(fbs), int(restecg), int(thalach), int(exang),
                float(oldpeak), int(slope), int(ca), int(thal)
            ]
            pred = heart_model.predict([user_input])
            st.session_state.heart_diagnosis = "Heart Disease" if pred[0] == 1 else "No Heart Disease"
        except:
            st.session_state.heart_diagnosis = "⚠️ Invalid input"

    if st.session_state.heart_diagnosis:
        st.success(f"Diagnosis: {st.session_state.heart_diagnosis}")

        st.subheader("💬 Heart Disease Assistant Chatbot")
        for msg in st.session_state.chat_heart:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("📋 Summary of Condition"):
            summary_text = get_chatbot_response("Heart Disease", st.session_state.heart_diagnosis, summary=True)
            st.info(summary_text)

        if user_query := st.chat_input("Ask about your heart condition..."):
            st.session_state.chat_heart.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Heart Disease", st.session_state.heart_diagnosis, user_query)
            st.session_state.chat_heart.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)

# =========================
# Parkinson's Prediction Page
# =========================
if selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: fo = st.text_input("MDVP:Fo")
    with col2: fhi = st.text_input("MDVP:Fhi")
    with col3: flo = st.text_input("MDVP:Flo")
    with col4: Jitter_percent = st.text_input("MDVP:Jitter %")
    with col5: Jitter_Abs = st.text_input("MDVP:Jitter Abs")
    with col1: RAP = st.text_input("MDVP:RAP")
    with col2: PPQ = st.text_input("MDVP:PPQ")
    with col3: DDP = st.text_input("Jitter:DDP")
    with col4: Shimmer = st.text_input("MDVP:Shimmer")
    with col5: Shimmer_dB = st.text_input("MDVP:Shimmer dB")
    with col1: APQ3 = st.text_input("Shimmer:APQ3")
    with col2: APQ5 = st.text_input("Shimmer:APQ5")
    with col3: APQ = st.text_input("MDVP:APQ")
    with col4: DDA = st.text_input("Shimmer:DDA")
    with col5: NHR = st.text_input("NHR")
    with col1: HNR = st.text_input("HNR")
    with col2: RPDE = st.text_input("RPDE")
    with col3: DFA = st.text_input("DFA")
    with col4: spread1 = st.text_input("Spread1")
    with col5: spread2 = st.text_input("Spread2")
    with col1: D2 = st.text_input("D2")
    with col2: PPE = st.text_input("PPE")

    if "parkinsons_diagnosis" not in st.session_state:
        st.session_state.parkinsons_diagnosis = ""
    if "chat_parkinsons" not in st.session_state:
        st.session_state.chat_parkinsons = []

    if st.button("🔍 Get Parkinson's Test Result"):
        try:
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent),
                float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]
            pred = parkinsons_model.predict([user_input])
            st.session_state.parkinsons_diagnosis = "Parkinson's Disease" if pred[0] == 1 else "No Parkinson's"
        except:
            st.session_state.parkinsons_diagnosis = "⚠️ Invalid input"

    if st.session_state.parkinsons_diagnosis:
        st.success(f"Diagnosis: {st.session_state.parkinsons_diagnosis}")

        st.subheader("💬 Parkinson's Assistant Chatbot")
        for msg in st.session_state.chat_parkinsons:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("📋 Summary of Condition"):
            summary_text = get_chatbot_response("Parkinson's Disease", st.session_state.parkinsons_diagnosis, summary=True)
            st.info(summary_text)

        if user_query := st.chat_input("Ask about Parkinson's condition..."):
            st.session_state.chat_parkinsons.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Parkinson's Disease", st.session_state.parkinsons_diagnosis, user_query)
            st.session_state.chat_parkinsons.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)
