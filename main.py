# -*- coding: utf-8 -*-
"""
Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
OpenAI (primary) + Gemini fallback (gemini-2.5-flash-lite)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
import google.generativeai as genai

# ---------------------------------------------------------
# 1️⃣ Load ML Models
# ---------------------------------------------------------
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# ---------------------------------------------------------
# 2️⃣ Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")

# ---------------------------------------------------------
# 3️⃣ Sidebar Menu
# ---------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction',
         'Parkinson’s Prediction', 'HealthBot Assistant'],
        icons=['activity', 'heart', 'brain', 'robot'],
        default_index=0
    )

# ---------------------------------------------------------
# 5️⃣ Diabetes Prediction
# ---------------------------------------------------------
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction using ML")

    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Blood Pressure value")
    SkinThickness = st.number_input("Skin Thickness value")
    Insulin = st.number_input("Insulin Level")
    BMI = st.number_input("BMI value")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value")
    Age = st.number_input("Age")

    if st.button('Diabetes Test Result'):
        user_input_d = [int(Pregnancies), Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diab_prediction = diabetes_model.predict([user_input_d])
        if diab_prediction[0] == 1:
            st.error('The person is likely to have diabetes.')
            diab_status = 'likely to have diabetes'
        else:
            st.success('The person is not diabetic.')
            diab_status = 'not diabetic'
        st.session_state['last_prediction'] = {
            'disease': 'Diabetes',
            'input': user_input_d,
            'result': diab_status
        }

# ---------------------------------------------------------
# 6️⃣ Heart Disease Prediction
# ---------------------------------------------------------
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 0)
        sex = st.selectbox('Sex (1=Male, 0=Female)', [0, 1])
        cp = st.number_input('Chest Pain types', 0)
        trestbps = st.number_input('Resting Blood Pressure', 0)
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', 0)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0, 1])
        restecg = st.number_input('Resting Electrocardiographic results', 0)
        thalach = st.number_input('Maximum Heart Rate achieved', 0)
    with col3:
        exang = st.selectbox('Exercise Induced Angina (1=True, 0=False)', [0, 1])
        oldpeak = st.number_input('ST depression induced by exercise', 0.0)
        slope = st.number_input('Slope of the peak exercise ST segment', 0)
        ca = st.number_input('Major vessels colored by fluoroscopy', 0)
        thal = st.number_input('thal (0=Normal, 1=Fixed defect, 2=Reversable defect)', 0)

    if st.button('Heart Disease Test Result'):
        user_input_h = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = heart_model.predict([user_input_h])
        if heart_prediction[0] == 1:
            st.error('The person is likely to have heart disease.')
            heart_status = 'likely to have heart disease'
        else:
            st.success('The person does not have any heart disease.')
            heart_status = 'does not have any heart disease'
        st.session_state['last_prediction'] = {
            'disease': 'Heart Disease',
            'input': user_input_h,
            'result': heart_status
        }

# ---------------------------------------------------------
# 7️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
if selected == 'Parkinson’s Prediction':
    st.title("Parkinson’s Disease Prediction using ML")

    inputs = []
    for i in range(1, 23):
        inputs.append(st.number_input(f'Feature {i}', 0.0))

    if st.button('Parkinson’s Test Result'):
        user_input_p = inputs
        park_prediction = parkinsons_model.predict([user_input_p])
        if park_prediction[0] == 1:
            st.error('The person likely has Parkinson’s Disease.')
            park_status = 'likely to have Parkinson’s Disease'
        else:
            st.success('The person is healthy.')
            park_status = 'does not have Parkinson’s Disease'
        st.session_state['last_prediction'] = {
            'disease': 'Parkinson’s Disease',
            'input': user_input_p,
            'result': park_status
        }

# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant (ChatGPT-like UI)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant (Gemini-only, ChatGPT-style layout)
# ---------------------------------------------------------
if selected == '🤖 HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    import google.generativeai as genai

    # ✅ Configure Gemini
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    except Exception as e:
        st.error("⚠️ Gemini API key missing or invalid. Cannot start chatbot.")
        st.stop()

    # ✅ Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ Chat message display (oldest → newest)
    chat_container = st.container()
    input_container = st.container()

    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='background-color:#2e2e2e;border-radius:12px;padding:10px;margin:6px 0;text-align:left;'>
                    <b>🧑 You:</b> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#1a3d5d;border-radius:12px;padding:10px;margin:6px 0;text-align:left;'>
                    <b>🤖 HealthBot:</b> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

    # ✅ Sticky bottom input
    with input_container:
        st.markdown("---")
        user_input = st.text_area("💬 Type your message:", key="chat_input", height=80, placeholder="Ask about health, fitness, nutrition, or wellness...")
        send_btn = st.button("Send", use_container_width=True)

    if send_btn and user_input.strip():
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Context and system prompt
        system_prompt = (
            "You are a professional, friendly AI health assistant named HealthBot. "
            "Your job is to give safe, general health and wellness guidance. "
            "You can explain medical terms, healthy habits, nutrition, and exercise tips. "
            "Do NOT give prescriptions, diagnoses, or specific treatments. "
            "If something sounds serious, advise seeing a doctor."
        )

        # Include any prediction context if available
        last_pred = st.session_state.get("last_prediction", None)
        user_context = ""
        if last_pred:
            user_context = (
                f"\n\nUser recently tested for {last_pred['disease']}.\n"
                f"Input data: {last_pred['input']}\n"
                f"Prediction result: {last_pred['result']}"
            )

        # Full prompt
        full_prompt = system_prompt + user_context + "\n\nUser Question: " + user_input

        # ✅ Gemini response
        try:
            gemini_response = gemini_model.generate_content(full_prompt)
            reply = gemini_response.text
        except Exception as ge:
            reply = f"⚠️ Gemini API error: {ge}"

        # Add bot reply
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # ✅ Auto refresh to show latest message
        st.rerun()






