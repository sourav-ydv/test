# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:00:19 2025
Updated version - Disease Chatbot Stable Edition
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
    return pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=200)

chatbot = load_chatbot()


def get_chatbot_response(disease_name, diagnosis, user_query=None, summary=False):
    """Generate chatbot responses about the detected disease."""
    if summary:
        prompt = f"Summarize the condition '{disease_name}' in simple medical language with precautions, symptoms, and lifestyle advice. Add a note to consult a doctor."
    else:
        prompt = (
            f"You are a medical assistant chatbot. The diagnosed condition is {disease_name}. "
            f"The diagnosis result is: {diagnosis}. "
            f"The user asked: '{user_query}'. "
            f"Answer only about {disease_name}. If the question is unrelated, politely refuse."
        )

    response = chatbot(prompt)[0]["generated_text"]
    return response.strip()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Disease Prediction & Chatbot", layout="wide")
st.title("🧠 Multi-Disease Prediction System + Chatbot Assistant")

with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson\'s Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ========== Diabetes Section ==========
if selected == 'Diabetes Prediction':
    st.header("🔍 Diabetes Prediction")

    # Load model
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''
    if st.button('🔎 Predict Diabetes'):
        try:
            user_input = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diabetes_prediction = diabetes_model.predict([user_input])
            diagnosis = "The person has Diabetes" if diabetes_prediction[0] == 1 else "The person does not have Diabetes"
        except ValueError:
            diagnosis = "⚠️ Please enter valid numeric values."

        st.success(diagnosis)
        st.session_state['disease_name'] = "Diabetes"
        st.session_state['diagnosis'] = diagnosis

# ========== Heart Disease Section ==========
if selected == 'Heart Disease Prediction':
    st.header("❤️ Heart Disease Prediction")

    heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

    age = st.text_input('Age')
    sex = st.text_input('Sex (1 = Male, 0 = Female)')
    cp = st.text_input('Chest Pain types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    ca = st.text_input('Major vessels colored by fluoroscopy')
    thal = st.text_input('Thal: 0 = Normal; 1 = Fixed defect; 2 = Reversible defect')

    heart_diagnosis = ''
    if st.button('🔎 Predict Heart Disease'):
        try:
            user_input = [
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg), float(thalach),
                float(exang), float(oldpeak), float(slope), float(ca), float(thal)
            ]
            heart_prediction = heart_disease_model.predict([user_input])
            heart_diagnosis = "The person has Heart Disease" if heart_prediction[0] == 1 else "The person does not have Heart Disease"
        except ValueError:
            heart_diagnosis = "⚠️ Please enter valid numeric values."

        st.success(heart_diagnosis)
        st.session_state['disease_name'] = "Heart Disease"
        st.session_state['diagnosis'] = heart_diagnosis

# ========== Parkinson’s Section ==========
if selected == "Parkinson's Prediction":
    st.header("🧩 Parkinson's Disease Prediction")

    parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

    # Simplified Input Fields
    Fo = st.text_input('Average Vocal Fundamental Frequency (Fo)')
    Fhi = st.text_input('Maximum Vocal Fundamental Frequency (Fhi)')
    Flo = st.text_input('Minimum Vocal Fundamental Frequency (Flo)')
    Jitter_percent = st.text_input('Jitter (%)')
    Shimmer = st.text_input('Shimmer')
    HNR = st.text_input('HNR')

    parkinsons_diagnosis = ''
    if st.button("🔎 Predict Parkinson's Disease"):
        try:
            user_input = [float(Fo), float(Fhi), float(Flo), float(Jitter_percent), float(Shimmer), float(HNR)]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = "The person has Parkinson's Disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's Disease"
        except ValueError:
            parkinsons_diagnosis = "⚠️ Please enter valid numeric values."

        st.success(parkinsons_diagnosis)
        st.session_state['disease_name'] = "Parkinson's Disease"
        st.session_state['diagnosis'] = parkinsons_diagnosis

# =========================
# Chatbot Section (after prediction)
# =========================
if 'diagnosis' in st.session_state:
    st.divider()
    st.subheader(f"💬 {st.session_state['disease_name']} Assistant Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if st.button("📝 Summary of Condition", key="summary_btn"):
        summary_text = get_chatbot_response(
            st.session_state['disease_name'],
            st.session_state['diagnosis'],
            summary=True
        )
        st.session_state.chat_history.append({"role": "assistant", "content": summary_text})
        st.chat_message("assistant").write(summary_text)

    if user_query := st.chat_input("Ask about your condition..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        bot_reply = get_chatbot_response(
            st.session_state['disease_name'],
            st.session_state['diagnosis'],
            user_query=user_query
        )
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)
