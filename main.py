# app.py
# -*- coding: utf-8 -*-
"""
Multiple Disease Prediction System with Persistent AI Chatbot
Fixed: persist session state, stable chatbot responses, safer OpenAI usage
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import openai
from pathlib import Path

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# ----- Helper: load model safely -----
def safe_load_model(path):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        st.error(f"Model file not found: {path}")
        return None
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

# Load models (paths relative to app root)
diabetes_model = safe_load_model("diabetes_model.sav")
heart_disease_model = safe_load_model("heart_disease_model.sav")
parkinsons_model = safe_load_model("parkinsons_model.sav")

# OpenAI API key (store securely in Streamlit secrets)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    # If running locally, allow user to paste key (optional)
    openai.api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")

# ----- Sidebar menu -----
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Initialize session_state storage for each module
if "diab_chat" not in st.session_state:
    st.session_state.diab_chat = []
if "heart_chat" not in st.session_state:
    st.session_state.heart_chat = []
if "parkinsons_chat" not in st.session_state:
    st.session_state.parkinsons_chat = []

if "diab_diagnosis" not in st.session_state:
    st.session_state.diab_diagnosis = ""
if "heart_diagnosis" not in st.session_state:
    st.session_state.heart_diagnosis = ""
if "parkinsons_diagnosis" not in st.session_state:
    st.session_state.parkinsons_diagnosis = ""

# Generic function to contact OpenAI safely
def ask_openai_system(prompt_user, system_instructions, max_tokens=250, temp=0.7):
    if not openai.api_key:
        return "⚠️ OpenAI API key not configured."
    try:
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt_user}
        ]
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error contacting OpenAI API: {e}"

# ------------------ DIABETES ------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    # numeric inputs with keys so Streamlit retains values across reruns
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=50, value=0, step=1, key="pregnancies")
    with col2:
        glucose = st.number_input('Glucose Level', min_value=0, max_value=500, value=120, step=1, key="glucose")
    with col3:
        blood_pressure = st.number_input('Blood Pressure value', min_value=0, max_value=300, value=70, step=1, key="blood_pressure")
    with col1:
        skin_thickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=20, step=1, key="skin_thickness")
    with col2:
        insulin = st.number_input('Insulin Level', min_value=0, max_value=1000, value=80, step=1, key="insulin")
    with col3:
        bmi = st.number_input('BMI value', min_value=0.0, max_value=100.0, value=25.0, format="%.2f", key="bmi")
    with col1:
        dpf = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=10.0, value=0.5, format="%.4f", key="dpf")
    with col2:
        age = st.number_input('Age of the Person', min_value=0, max_value=150, value=30, step=1, key="age_diab")

    if st.button('Diabetes Test Result'):
        if diabetes_model is None:
            st.session_state.diab_diagnosis = "Model unavailable."
        else:
            try:
                user_input_d = [
                    int(pregnancies), int(glucose), int(blood_pressure),
                    int(skin_thickness), int(insulin), float(bmi),
                    float(dpf), int(age)
                ]
                pred = diabetes_model.predict([user_input_d])[0]
                if pred == 1:
                    st.session_state.diab_diagnosis = 'The person is likely diabetic (ML prediction).'
                else:
                    st.session_state.diab_diagnosis = 'The person is likely NOT diabetic (ML prediction).'
                # store last input for context
                st.session_state.diab_last_input = user_input_d
            except Exception as e:
                st.session_state.diab_diagnosis = f"Error during prediction: {e}"

    st.success(st.session_state.diab_diagnosis)

    # ---------------- AI Chatbot ----------------
    st.subheader("💬 Diabetes HealthBot")
    with st.form(key="diab_form"):
        user_query = st.text_input("Ask about your diabetic condition", key="diab_query")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        # build prompt including last prediction if available
        last_input = st.session_state.get("diab_last_input", None)
        diagnosis = st.session_state.get("diab_diagnosis", "")
        context = ""
        if last_input is not None and diagnosis:
            context = f"User numeric inputs (Preg,Gluc,BP,Skin,Insulin,BMI,DPF,Age): {last_input}. Diagnosis summary: {diagnosis}."
        else:
            context = "No ML prediction available for context."

        prompt = (
            f"{context} Answer safely focusing on lifestyle, diet, exercise, and precautions for diabetes. "
            f"DO NOT provide prescriptions or dosing. If the question requires urgent medical attention, advise to seek immediate care. "
            f"User question: {user_query}"
        )

        system_instructions = (
            "You are a helpful, cautious medical-advice-style assistant. Provide evidence-aligned lifestyle/diet/exercise suggestions only. "
            "Do not give or suggest prescriptions, dosages, or definitive diagnoses. Always include a short disclaimer to consult a clinician for personal medical advice."
        )

        answer = ask_openai_system(prompt, system_instructions)
        st.session_state.diab_chat.append({"user": user_query, "bot": answer})

    for chat in st.session_state.diab_chat:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**HealthBot:** {chat['bot']}")

# ------------------ HEART DISEASE ------------------
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=150, value=45, key="age_h")
    with col2:
        sex = st.selectbox('Sex', options=[0,1], format_func=lambda x: "Female (0)" if x==0 else "Male (1)", key="sex_h")
    with col3:
        cp = st.number_input('Chest Pain types (0-3)', min_value=0, max_value=3, value=0, step=1, key="cp")
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120, key="trestbps")
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=1000, value=240, key="chol")
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0,1], key="fbs")
    with col1:
        restecg = st.number_input('Resting ECG results (0-2)', min_value=0, max_value=2, value=1, step=1, key="restecg")
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, max_value=300, value=150, key="thalach")
    with col3:
        exang = st.selectbox('Exercise Induced Angina (0/1)', options=[0,1], key="exang")
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, format="%.2f", key="oldpeak")
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment (0-2)', min_value=0, max_value=2, value=1, step=1, key="slope")
    with col3:
        ca = st.number_input('Major vessels colored by fluoroscopy (0-3)', min_value=0, max_value=3, value=0, step=1, key="ca")
    with col1:
        thal = st.number_input('thal: 0=normal;1=fixed;2=reversible', min_value=0, max_value=2, value=0, step=1, key="thal")

    if st.button('Heart Disease Test Result'):
        if heart_disease_model is None:
            st.session_state.heart_diagnosis = "Model unavailable."
        else:
            try:
                user_input_h = [
                    int(age), int(sex), int(cp), int(trestbps), int(chol),
                    int(fbs), int(restecg), int(thalach), int(exang),
                    float(oldpeak), int(slope), int(ca), int(thal)
                ]
                pred = heart_disease_model.predict([user_input_h])[0]
                if pred == 1:
                    st.session_state.heart_diagnosis = 'ML predicts presence of heart disease (possible).'
                else:
                    st.session_state.heart_diagnosis = 'ML predicts low probability of heart disease.'
                st.session_state.heart_last_input = user_input_h
            except Exception as e:
                st.session_state.heart_diagnosis = f"Error during prediction: {e}"

    st.success(st.session_state.heart_diagnosis)

    # Heart chatbot
    st.subheader("💬 Heart HealthBot")
    with st.form(key="heart_form"):
        user_query = st.text_input("Ask about your heart condition", key="heart_query")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        last_input = st.session_state.get("heart_last_input", None)
        diagnosis = st.session_state.get("heart_diagnosis", "")
        context = f"User numeric inputs: {last_input}. Diagnosis summary: {diagnosis}." if last_input else "No ML prediction available for context."

        prompt = (
            f"{context} Answer focusing on lifestyle, exercise, risk factor modification, and precautions for heart disease. "
            f"Do not give prescriptions. If urgent warning signs (chest pain, breathlessness, fainting) are described advise immediate medical attention. "
            f"User question: {user_query}"
        )
        system_instructions = (
            "You are a cautious medical assistant: provide general lifestyle/risk-factor guidance and clearly state limitations. "
            "Do NOT provide medical prescriptions or claims of definitive diagnosis. Encourage consultation with a physician for personalized advice."
        )

        answer = ask_openai_system(prompt, system_instructions)
        st.session_state.heart_chat.append({"user": user_query, "bot": answer})

    for chat in st.session_state.heart_chat:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**HealthBot:** {chat['bot']}")

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
        # each input has a unique key so state persists
        with [col1, col2, col3, col4, col5][i % 5]:
            val = st.number_input(field, value=0.0, format="%.6f", key=f"par_{field}")
            inputs.append(val)

    if st.button("Parkinson's Test Result"):
        if parkinsons_model is None:
            st.session_state.parkinsons_diagnosis = "Model unavailable."
        else:
            try:
                user_input_p = [float(x) for x in inputs]
                pred = parkinsons_model.predict([user_input_p])[0]
                if pred == 1:
                    st.session_state.parkinsons_diagnosis = "ML predicts Parkinson's disease is likely."
                else:
                    st.session_state.parkinsons_diagnosis = "ML predicts Parkinson's disease is unlikely."
                st.session_state.parkinsons_last_input = user_input_p
            except Exception as e:
                st.session_state.parkinsons_diagnosis = f"Error during prediction: {e}"

    st.success(st.session_state.parkinsons_diagnosis)

    # Parkinson's chatbot
    st.subheader("💬 Parkinson's HealthBot")
    with st.form(key="parkinsons_form"):
        user_query = st.text_input("Ask about Parkinson’s disease", key="par_query")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        last_input = st.session_state.get("parkinsons_last_input", None)
        diagnosis = st.session_state.get("parkinsons_diagnosis", "")
        context = f"User numeric inputs: {last_input}. Diagnosis summary: {diagnosis}." if last_input else "No ML prediction available for context."

        prompt = (
            f"{context} Answer focusing on lifestyle, therapy, exercise, and precautions for Parkinson's disease. "
            f"Do not provide prescriptions or medical dosing instructions. If urgent signs are mentioned, recommend clinician contact."
            f" User question: {user_query}"
        )
        system_instructions = (
            "You are a cautious assistant: provide supportive, evidence-informed guidance about non-prescription options and recommend clinicians for medical therapy."
        )
        answer = ask_openai_system(prompt, system_instructions)
        st.session_state.parkinsons_chat.append({"user": user_query, "bot": answer})

    for chat in st.session_state.parkinsons_chat:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**HealthBot:** {chat['bot']}")
