# -*- coding: utf-8 -*-
"""
Multiple Disease Prediction System with Persistent AI Chatbot
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import openai

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Load ML models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# OpenAI API key (store securely in .streamlit/secrets.toml)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------ SIDEBAR ------------------
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ------------------ DIABETES ------------------
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
    user_input_d = []

    if st.button('Diabetes Test Result'):
        try:
            user_input_d = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), int(Age)
            ]
            diab_prediction = diabetes_model.predict([user_input_d])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = "Please enter valid numeric values."

    st.success(diab_diagnosis)

    # ---------------- AI Chatbot ----------------
    st.subheader("💬 Diabetes HealthBot")
    if "diab_chat" not in st.session_state:
        st.session_state.diab_chat = []

    # Display chat history
    for chat in st.session_state.diab_chat:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**HealthBot:** {chat['bot']}")

    # Chat input form
    with st.form("diab_chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask about your diabetic condition")
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            prompt = (
                f"User inputs: {user_input_d}, Diagnosis: {diab_diagnosis}. "
                f"Answer the user's question safely, focusing on lifestyle, diet, exercise, and precautions. "
                f"Do not give medical prescriptions. Question: {user_query}"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.7
                )
                answer = response['choices'][0]['message']['content']
            except Exception:
                answer = "⚠️ Error contacting OpenAI API."

            st.session_state.diab_chat.append({"user": user_query, "bot": answer})
            st.experimental_rerun()


# ------------------ HEART DISEASE ------------------
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
    user_input_h = []

    if st.button('Heart Disease Test Result'):
        try:
            user_input_h = [
                int(age), int(sex), int(cp), int(trestbps), int(chol),
                int(fbs), int(restecg), int(thalach), int(exang),
                float(oldpeak), int(slope), int(ca), int(thal)
            ]
            heart_prediction = heart_disease_model.predict([user_input_h])
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        except ValueError:
            heart_diagnosis = "Please enter valid numeric values."

    st.success(heart_diagnosis)

    # ---------------- AI Chatbot ----------------
    st.subheader("💬 Heart HealthBot")
    if "heart_chat" not in st.session_state:
        st.session_state.heart_chat = []

    # Display chat history
    for chat in st.session_state.heart_chat:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**HealthBot:** {chat['bot']}")

    # Chat input form
    with st.form("heart_chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask about your heart condition")
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            prompt = (
                f"User inputs: {user_input_h}, Diagnosis: {heart_diagnosis}. "
                f"Answer the user's question safely, focusing on lifestyle, diet, exercise, and precautions. "
                f"Do not give medical prescriptions. Question: {user_query}"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.7
                )
                answer = response['choices'][0]['message']['content']
            except Exception:
                answer = "⚠️ Error contacting OpenAI API."

            st.session_state.heart_chat.append({"user": user_query, "bot": answer})
            st.experimental_rerun()


# ------------------ PARKINSONS ------------------
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
    user_input_p = []

    if st.button("Parkinson's Test Result"):
        try:
            user_input_p = [float(x) for x in inputs]
            parkinsons_prediction = parkinsons_model.predict([user_input_p])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = "⚠️ Please enter valid numeric values."

    st.success(parkinsons_diagnosis)

    # ---------------- AI Chatbot ----------------
    st.subheader("💬 Parkinson's HealthBot")
    if "parkinsons_chat" not in st.session_state:
        st.session_state.parkinsons_chat = []

    # Display chat history
    for chat in st.session_state.parkinsons_chat:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**HealthBot:** {chat['bot']}")

    # Chat input form
    with st.form("parkinsons_chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask about Parkinson’s disease")
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            prompt = (
                f"User inputs: {user_input_p}, Diagnosis: {parkinsons_diagnosis}. "
                f"Answer the user's question safely, focusing on lifestyle, exercise, therapy, and precautions. "
                f"Do not give medical prescriptions. Question: {user_query}"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.7
                )
                answer = response['choices'][0]['message']['content']
            except Exception:
                answer = "⚠️ Error contacting OpenAI API."

            st.session_state.parkinsons_chat.append({"user": user_query, "bot": answer})
            st.experimental_rerun()
