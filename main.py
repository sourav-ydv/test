# -*- coding: utf-8 -*-
"""
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
# Load free model (first run downloads ~1GB)
@st.cache_resource
def load_chatbot():
    return pipeline("text2text-generation", model="google/flan-t5-base")

chatbot = load_chatbot()

# =========================
# Helper: AI Chatbot
# =========================
def get_chatbot_response(disease, diagnosis, user_input, user_query):
    prompt = f"""
    You are a medical assistant chatbot. 
    The patient is being checked for {disease}.
    
    Model Diagnosis: {diagnosis}
    Patient Input Data: {user_input}

    The user is asking: {user_query}

    Based on the data and diagnosis, provide suggestions, precautions, 
    severity estimation if possible, and lifestyle modifications in simple language. 
    Always include a disclaimer to consult a doctor.
    """
    output = chatbot(prompt, max_length=256, do_sample=True)
    return output[0]['generated_text']

# =========================
# Load Saved Models
# =========================
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# =========================
# Diabetes Prediction Page
# =========================
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies')
    with col2: Glucose = st.text_input('Glucose Level (mg/dL)')
    with col3: BloodPressure = st.text_input('Blood Pressure (mm Hg)')
    with col1: SkinThickness = st.text_input('Skin Thickness (mm)')
    with col2: Insulin = st.text_input('Insulin Level (mu U/ml)')
    with col3: BMI = st.text_input('BMI (Body Mass Index)')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2: Age = st.text_input('Age (years)')

    diab_diagnosis = ''
    user_input_dict = {}

    if st.button('🔍 Get Diabetes Test Result'):
        try:
            user_input = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), int(Age)
            ]
            diab_prediction = diabetes_model.predict([user_input])
            user_input_dict = {
                "Pregnancies": Pregnancies, "Glucose": Glucose,
                "BloodPressure": BloodPressure, "SkinThickness": SkinThickness,
                "Insulin": Insulin, "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age
            }
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person **is Diabetic**'
            else:
                diab_diagnosis = 'The person **is Not Diabetic**'
        except ValueError:
            diab_diagnosis = "⚠️ Please enter valid numeric values."

        st.success(diab_diagnosis)

        # Chatbot Section (appears only after prediction)
        st.subheader("💬 Diabetes Assistant Chatbot")
        if "chat_diab" not in st.session_state: st.session_state.chat_diab = []
        for msg in st.session_state.chat_diab: st.chat_message(msg["role"]).write(msg["content"])
        if user_query := st.chat_input("Ask about your diabetes condition..."):
            st.session_state.chat_diab.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Diabetes", diab_diagnosis, user_input_dict, user_query)
            st.session_state.chat_diab.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)

# =========================
# Heart Disease Prediction Page
# =========================
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age (years)')
    with col2: sex = st.text_input('Sex (0 = Female, 1 = Male)')
    with col3: cp = st.text_input('Chest Pain Type (0–3)')
    with col1: trestbps = st.text_input('Resting Blood Pressure (mm Hg)')
    with col2: chol = st.text_input('Cholesterol (mg/dL)')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)')
    with col1: restecg = st.text_input('Resting ECG Results (0–2)')
    with col2: thalach = st.text_input('Max Heart Rate Achieved')
    with col3: exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    with col1: oldpeak = st.text_input('ST Depression (oldpeak)')
    with col2: slope = st.text_input('Slope of Peak Exercise ST (0–2)')
    with col3: ca = st.text_input('Major Vessels (0–3)')
    with col1: thal = st.text_input('Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)')

    heart_diagnosis = ''
    user_input_dict = {}

    if st.button('🔍 Get Heart Disease Test Result'):
        try:
            user_input = [
                int(age), int(sex), int(cp), int(trestbps), int(chol),
                int(fbs), int(restecg), int(thalach), int(exang),
                float(oldpeak), int(slope), int(ca), int(thal)
            ]
            heart_prediction = heart_disease_model.predict([user_input])
            user_input_dict = {
                "Age": age, "Sex": sex, "Chest Pain": cp,
                "Resting BP": trestbps, "Cholesterol": chol, "Fasting Blood Sugar": fbs,
                "Rest ECG": restecg, "Max HR": thalach, "Exercise Angina": exang,
                "Oldpeak": oldpeak, "Slope": slope, "CA": ca, "Thal": thal
            }
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person **has Heart Disease**'
            else:
                heart_diagnosis = 'The person **does not have Heart Disease**'
        except ValueError:
            heart_diagnosis = "⚠️ Please enter valid numeric values."

        st.success(heart_diagnosis)

        # Chatbot Section
        st.subheader("💬 Heart Disease Assistant Chatbot")
        if "chat_heart" not in st.session_state: st.session_state.chat_heart = []
        for msg in st.session_state.chat_heart: st.chat_message(msg["role"]).write(msg["content"])
        if user_query := st.chat_input("Ask about your heart condition..."):
            st.session_state.chat_heart.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Heart Disease", heart_diagnosis, user_input_dict, user_query)
            st.session_state.chat_heart.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)

# =========================
# Parkinson's Prediction Page
# =========================
if selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: fo = st.text_input('MDVP:Fo (Hz)')
    with col2: fhi = st.text_input('MDVP:Fhi (Hz)')
    with col3: flo = st.text_input('MDVP:Flo (Hz)')
    with col4: Jitter_percent = st.text_input('MDVP:Jitter (%)')
    with col5: Jitter_Abs = st.text_input('MDVP:Jitter (Abs)')
    with col1: RAP = st.text_input('MDVP:RAP')
    with col2: PPQ = st.text_input('MDVP:PPQ')
    with col3: DDP = st.text_input('Jitter:DDP')
    with col4: Shimmer = st.text_input('MDVP:Shimmer')
    with col5: Shimmer_dB = st.text_input('MDVP:Shimmer (dB)')
    with col1: APQ3 = st.text_input('Shimmer:APQ3')
    with col2: APQ5 = st.text_input('Shimmer:APQ5')
    with col3: APQ = st.text_input('MDVP:APQ')
    with col4: DDA = st.text_input('Shimmer:DDA')
    with col5: NHR = st.text_input('NHR')
    with col1: HNR = st.text_input('HNR')
    with col2: RPDE = st.text_input('RPDE')
    with col3: DFA = st.text_input('DFA')
    with col4: spread1 = st.text_input('Spread1')
    with col5: spread2 = st.text_input('Spread2')
    with col1: D2 = st.text_input('D2')
    with col2: PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    user_input_dict = {}

    if st.button("🔍 Get Parkinson's Test Result"):
        try:
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent),
                float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            user_input_dict = {
                "Fo": fo, "Fhi": fhi, "Flo": flo, "Jitter %": Jitter_percent,
                "Jitter Abs": Jitter_Abs, "RAP": RAP, "PPQ": PPQ, "DDP": DDP,
                "Shimmer": Shimmer, "Shimmer dB": Shimmer_dB,
                "APQ3": APQ3, "APQ5": APQ5, "APQ": APQ, "DDA": DDA,
                "NHR": NHR, "HNR": HNR, "RPDE": RPDE, "DFA": DFA,
                "Spread1": spread1, "Spread2": spread2, "D2": D2, "PPE": PPE
            }
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person **has Parkinson's Disease**"
            else:
                parkinsons_diagnosis = "The person **does not have Parkinson's Disease**"
        except ValueError:
            parkinsons_diagnosis = "⚠️ Please enter valid numeric values."

        st.success(parkinsons_diagnosis)

        # Chatbot Section
        st.subheader("💬 Parkinson's Assistant Chatbot")
        if "chat_parkinsons" not in st.session_state: st.session_state.chat_parkinsons = []
        for msg in st.session_state.chat_parkinsons: st.chat_message(msg["role"]).write(msg["content"])
        if user_query := st.chat_input("Ask about Parkinson's condition..."):
            st.session_state.chat_parkinsons.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Parkinson's Disease", parkinsons_diagnosis, user_input_dict, user_query)
            st.session_state.chat_parkinsons.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)
