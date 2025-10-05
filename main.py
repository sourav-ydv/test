# -*- coding: utf-8 -*-
"""
Disease Prediction + Smart Medical Chatbot (Hugging Face version)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from transformers import pipeline

# =========================
# Load ML Models
# =========================
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# =========================
# Hugging Face Chatbot Setup (Text Generation)
# =========================
chatbot_pipeline = pipeline(
    "text-generation",
    model="facebook/blenderbot-400M-distill"
)

# =========================
# Sidebar Menu
# =========================
with st.sidebar:
    selected = option_menu(
        'Disease Prediction & Assistant System',
        ['Diabetes', 'Heart Disease', 'Parkinsons'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# =========================
# Helper Functions
# =========================
def get_prediction(model, input_data):
    try:
        prediction = model.predict([input_data])
        return prediction[0]
    except Exception:
        return "Error"

def chatbot_response(disease, user_input):
    unrelated = ["president", "game", "movie", "AI", "chatbot", "politics", "sports"]
    if any(word in user_input.lower() for word in unrelated):
        return "❌ I can only discuss medical information related to your disease."

    prompt = (
        f"You are a medical assistant chatbot helping a patient diagnosed with {disease}. "
        f"The user asked: '{user_input}'. "
        f"Provide accurate, health-related guidance — symptoms, precautions, lifestyle, and treatment advice. "
        f"Always add: 'Consult your doctor for personalized care.'"
    )

    try:
        response = chatbot_pipeline(prompt, max_new_tokens=150, num_return_sequences=1)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

def summarize_condition(disease):
    prompt = f"Summarize the medical condition '{disease}' with symptoms, prevention, and lifestyle advice in simple words."
    try:
        response = chatbot_pipeline(prompt, max_new_tokens=120, num_return_sequences=1)[0]["generated_text"]
        return response.strip()
    except:
        return f"{disease} is a health condition. Please consult your doctor for more information."

# =========================
# Diabetes Prediction Page
# =========================
if selected == 'Diabetes':
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

    diagnosis = ''

    if st.button('🔍 Predict Diabetes Result'):
        try:
            user_input = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            result = get_prediction(diabetes_model, user_input)
            diagnosis = 'Diabetic' if result == 1 else 'Not Diabetic'
        except:
            diagnosis = 'Invalid input data.'

        st.success(f'The person is {diagnosis}')

        # Chatbot
        st.subheader("💬 Diabetes Assistant Chatbot")
        if st.button("📄 Summary of Condition"):
            st.info(summarize_condition("Diabetes"))

        user_query = st.text_input("Ask about your diabetes condition:")
        if user_query:
            reply = chatbot_response("Diabetes", user_query)
            st.markdown(f"**🤖 Chatbot:** {reply}")

# =========================
# Heart Disease Prediction Page
# =========================
if selected == 'Heart Disease':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age')
    with col2: sex = st.text_input('Sex (1 = male, 0 = female)')
    with col3: cp = st.text_input('Chest Pain Type')
    with col1: trestbps = st.text_input('Resting Blood Pressure')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)')
    with col1: restecg = st.text_input('Resting Electrocardiographic results')
    with col2: thalach = st.text_input('Maximum Heart Rate achieved')
    with col3: exang = st.text_input('Exercise Induced Angina (1 = yes, 0 = no)')
    with col1: oldpeak = st.text_input('ST depression induced by exercise')
    with col2: slope = st.text_input('Slope of the peak exercise ST segment')
    with col3: ca = st.text_input('Major vessels colored by fluoroscopy')
    with col1: thal = st.text_input('Thal (0 = normal; 1 = fixed defect; 2 = reversable defect)')

    heart_diagnosis = ''

    if st.button('🔍 Predict Heart Disease Result'):
        try:
            user_input = [
                float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                float(restecg), float(thalach), float(exang), float(oldpeak),
                float(slope), float(ca), float(thal)
            ]
            result = get_prediction(heart_disease_model, user_input)
            heart_diagnosis = 'Heart Disease Detected' if result == 1 else 'No Heart Disease'
        except:
            heart_diagnosis = 'Invalid input data.'

        st.success(f'The person is {heart_diagnosis}')

        st.subheader("💬 Heart Disease Assistant Chatbot")
        if st.button("📄 Summary of Condition"):
            st.info(summarize_condition("Heart Disease"))

        user_query = st.text_input("Ask about your heart condition:")
        if user_query:
            reply = chatbot_response("Heart Disease", user_query)
            st.markdown(f"**🤖 Chatbot:** {reply}")

# =========================
# Parkinson’s Prediction Page
# =========================
if selected == "Parkinsons":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    inputs = [st.text_input(name) for name in features]
    parkinsons_diagnosis = ''

    if st.button("🔍 Predict Parkinson's Result"):
        try:
            user_input = [float(i) for i in inputs]
            result = get_prediction(parkinsons_model, user_input)
            parkinsons_diagnosis = "Parkinson's Detected" if result == 1 else "No Parkinson's"
        except:
            parkinsons_diagnosis = 'Invalid input data.'

        st.success(f'The person is {parkinsons_diagnosis}')

        st.subheader("💬 Parkinson’s Assistant Chatbot")
        if st.button("📄 Summary of Condition"):
            st.info(summarize_condition("Parkinson's Disease"))

        user_query = st.text_input("Ask about your Parkinson's condition:")
        if user_query:
            reply = chatbot_response("Parkinson's Disease", user_query)
            st.markdown(f"**🤖 Chatbot:** {reply}")
