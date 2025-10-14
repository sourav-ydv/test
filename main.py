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
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age")

    if st.button('Diabetes Test Result'):
        user_input_d = [int(Pregnancies), int(Glucose), int(BloodPressure),
                      int(SkinThickness), int(Insulin), float(BMI),
                      float(DiabetesPedigreeFunction), int(Age)]
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
        age = st.text_input('Age')
        sex = st.text_input('Sex (1=Male, 0=Female)')
        cp = st.text_input('Chest Pain types')
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)')
        restecg = st.text_input('Resting Electrocardiographic results')
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=True, 0=False)')
        oldpeak = st.text_input('ST depression induced by exercise')
        slope = st.text_input('Slope of the peak exercise ST segment')
        ca = st.text_input('Major vessels colored by fluoroscopy')
        thal = st.text_input('thal (0=Normal, 1=Fixed defect, 2=Reversable defect)')

    if st.button('Heart Disease Test Result'):
        user_input_h = [
            int(age), int(sex), int(cp), int(trestbps), int(chol),
            int(fbs), int(restecg), int(thalach), int(exang),
            float(oldpeak), int(slope), int(ca), int(thal)
        ]
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
# 8️⃣ HealthBot Assistant (Gemini-Only Chatbot with Enter-to-Send)
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    # --- Gemini Setup ---
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("⚠️ Gemini API key missing or invalid. Please check your configuration.")
        st.stop()

    # --- Initialize Chat Memory ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # --- Chat Display Container ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#1e1e1e;padding:10px 15px;border-radius:12px;"
                    f"margin:8px 0;text-align:right;color:#fff;'>"
                    f"🧑 <b>You:</b> {msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#2b313e;padding:10px 15px;border-radius:12px;"
                    f"margin:8px 0;text-align:left;color:#e2e2e2;'>"
                    f"🤖 <b>HealthBot:</b> {msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # --- Function: Send Message ---
    def handle_send():
        user_text = st.session_state.chat_input.strip()
        if not user_text:
            return

        st.session_state.chat_history.append({"role": "user", "content": user_text})

        # --- System Prompt ---
        system_prompt = (
            "You are a helpful and knowledgeable AI health assistant named HealthBot. "
            "Provide general information on health, wellness, exercise, and diet. "
            "Avoid giving any medical prescriptions or diagnoses. "
            "Encourage users to consult professionals for medical concerns."
        )

        # --- Add Disease Prediction Context ---
        last_pred = st.session_state.get('last_prediction', None)
        user_context = ""
        if last_pred:
            disease = last_pred['disease']
            values = last_pred['input']

            if disease == "Diabetes":
                columns = [
                    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
                ]
            elif disease == "Heart Disease":
                columns = [
                    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                    "FastingBS", "RestECG", "MaxHR", "ExerciseAngina",
                    "Oldpeak", "Slope", "CA", "Thal"
                ]
            elif disease == "Parkinson’s Disease":
                columns = [f"Feature_{i}" for i in range(1, 23)]
            else:
                columns = []

            input_with_names = "\n".join([f"{c}: {v}" for c, v in zip(columns, values)])

            user_context = (
                f"\nUser recently tested for {disease}.\n"
                f"Input details:\n{input_with_names}\n"
                f"Prediction result: {last_pred['result']}\n"
                "Give lifestyle advice, diet tips, or precautions related to this data."
            )

        full_prompt = f"{system_prompt}\n{user_context}\n\nUser Question: {user_text}"

        # --- Gemini Response ---
        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(full_prompt)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Gemini API error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # ✅ Clear input safely
        st.session_state.chat_input = ""

    # --- Fixed Input Box at Bottom with Enter-to-Send ---
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 80px !important;
        }
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #111;
            padding: 12px;
            border-top: 1px solid #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        user_input = st.text_area(
            "💬 Type your message...",
            key="chat_input",
            height=80,
            placeholder="Ask about diet, fitness, or your health data...",
            label_visibility="collapsed"
        )

        # Detect Enter key press
        if user_input and st.session_state.chat_input == user_input:
            handle_send()

    # Optional Clear Chat button
    st.button("🧹 Clear Chat", use_container_width=True, on_click=lambda: st.session_state.update({"chat_history": [], "chat_input": ""}))

