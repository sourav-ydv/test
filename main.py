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

# ------------------ LOAD MODELS ------------------
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

# ------------------ HELPER FUNCTION FOR CHAT ------------------
def chat_interface(chat_key, user_input_summary, bot_topic, user_query_input_key):
    """Display chat interface with persistent session state for a disease."""
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.subheader(f"💬 {bot_topic} HealthBot")

    # Display chat history
    st.markdown('<div style="max-height:300px;overflow-y:auto;">', unsafe_allow_html=True)
    for chat in st.session_state[chat_key]:
        st.markdown(f"<div style='background:#DCF8C6;padding:5px;border-radius:10px;margin:5px;text-align:right'>{chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#E8E8E8;padding:5px;border-radius:10px;margin:5px;text-align:left'>{chat['bot']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input form
    with st.form(f"{chat_key}_form", clear_on_submit=True):
        user_query = st.text_input("Ask your question...", key=user_query_input_key)
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            prompt = (
                f"User inputs: {user_input_summary}. "
                f"Answer safely, focusing on lifestyle, diet, exercise, and precautions. "
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

            st.session_state[chat_key].append({"user": user_query, "bot": answer})
            st.experimental_rerun()


# ------------------ DIABETES ------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies')
    with col2: Glucose = st.text_input('Glucose Level')
    with col3: BloodPressure = st.text_input('Blood Pressure value')
    with col1: SkinThickness = st.text_input('Skin Thickness value')
    with col2: Insulin = st.text_input('Insulin Level')
    with col3: BMI = st.text_input('BMI value')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2: Age = st.text_input('Age of the Person')

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

    if diab_diagnosis:
        chat_interface(
            chat_key="diab_chat",
            user_input_summary=f"{user_input}, Diagnosis: {diab_diagnosis}",
            bot_topic="Diabetes",
            user_query_input_key="diab_input"
        )


# ------------------ HEART DISEASE ------------------
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age')
    with col2: sex = st.text_input('Sex')
    with col3: cp = st.text_input('Chest Pain types')
    with col1: trestbps = st.text_input('Resting Blood Pressure')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1: restecg = st.text_input('Resting Electrocardiographic results')
    with col2: thalach = st.text_input('Maximum Heart Rate achieved')
    with col3: exang = st.text_input('Exercise Induced Angina')
    with col1: oldpeak = st.text_input('ST depression induced by exercise')
    with col2: slope = st.text_input('Slope of the peak exercise ST segment')
    with col3: ca = st.text_input('Major vessels colored by flourosopy')
    with col1: thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

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

    if heart_diagnosis:
        chat_interface(
            chat_key="heart_chat",
            user_input_summary=f"{user_input}, Diagnosis: {heart_diagnosis}",
            bot_topic="Heart",
            user_query_input_key="heart_input"
        )


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

    if parkinsons_diagnosis:
        chat_interface(
            chat_key="parkinsons_chat",
            user_input_summary=f"{user_input}, Diagnosis: {parkinsons_diagnosis}",
            bot_topic="Parkinson's",
            user_query_input_key="parkinsons_input"
        )
