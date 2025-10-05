# -*- coding: utf-8 -*-
"""
Disease Prediction + Smart Medical Chatbot
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
# Chatbot Setup (Hugging Face)
# =========================
chatbot_pipeline = pipeline("conversational", model="facebook/blenderbot-400M-distill")

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
    if disease not in ["Diabetes", "Heart Disease", "Parkinson's"]:
        return "Error: Invalid disease context."

    unrelated = ["president", "country", "movie", "game", "politics", "AI", "chatbot"]
    if any(word in user_input.lower() for word in unrelated):
        return "❌ Sorry, I can only answer questions related to your medical condition."

    prompt = (
        f"You are a medical assistant chatbot. The user has {disease}. "
        f"Question: {user_input}. "
        f"Give clear, factual, health-related advice with symptoms, treatment options, and lifestyle tips. "
        f"Always include a disclaimer to consult a doctor."
    )

    try:
        response = chatbot_pipeline(prompt)[0]['generated_text']
        return response
    except Exception:
        return "⚠️ Sorry, I couldn’t process your question. Please try again."

def summarize_condition(disease):
    prompt = f"Summarize the condition {disease} in simple language with symptoms, precautions, lifestyle changes, and doctor advice."
    try:
        response = chatbot_pipeline(prompt)[0]['generated_text']
        return response
    except Exception:
        return f"{disease} is a medical condition. Please consult your doctor for more details."

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

        # Chatbot after prediction
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

    col1, col2, col3 = st.columns(3)
    with col1: fo = st.text_input('MDVP:Fo(Hz)')
    with col2: fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3: flo = st.text_input('MDVP:Flo(Hz)')
    with col1: Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col2: Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col3: RAP = st.text_input('MDVP:RAP')
    with col1: PPQ = st.text_input('MDVP:PPQ')
    with col2: DDP = st.text_input('Jitter:DDP')
    with col3: Shimmer = st.text_input('MDVP:Shimmer')
    with col1: Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col2: APQ3 = st.text_input('Shimmer:APQ3')
    with col3: APQ5 = st.text_input('Shimmer:APQ5')
    with col1: APQ = st.text_input('MDVP:APQ')
    with col2: DDA = st.text_input('Shimmer:DDA')
    with col3: NHR = st.text_input('NHR')
    with col1: HNR = st.text_input('HNR')
    with col2: RPDE = st.text_input('RPDE')
    with col3: DFA = st.text_input('DFA')
    with col1: spread1 = st.text_input('spread1')
    with col2: spread2 = st.text_input('spread2')
    with col3: D2 = st.text_input('D2')
    with col1: PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button("🔍 Predict Parkinson's Result"):
        try:
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]
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
