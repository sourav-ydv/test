import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from transformers import pipeline

# =========================
# Hugging Face Chatbot Setup
# =========================
@st.cache_resource
def load_chatbot():
    # Use a text2text-generation model for instruction-following
    return pipeline("text2text-generation", model="google/flan-t5-small")

chatbot = load_chatbot()

def get_chatbot_response(disease_name, diagnosis, user_query):
    """
    Generate response using Hugging Face Flan-T5.
    Strictly answer only disease-related queries.
    """
    # If query is unrelated to disease
    if not any(word in user_query.lower() for word in [disease_name.lower(), "symptom", "treatment", "diet", "exercise", "doctor", "medicine", "condition", "health", "cure", "prevent", "risk"]):
        return f"I can only answer questions about {disease_name}. Please ask about symptoms, treatments, lifestyle, or precautions."

    # Construct disease-specific prompt
    prompt = (
        f"You are a helpful medical assistant. The user has {disease_name}. "
        f"Diagnosis: {diagnosis}. "
        f"The user asked: {user_query}\n\n"
        f"Provide a clear, helpful answer including suggestions, lifestyle tips, and a disclaimer to consult a doctor."
    )

    response = chatbot(prompt, max_length=200)
    return response[0]['generated_text'].replace(prompt, "").strip()

# =========================
# Load Saved Models
# =========================
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.sav", "rb"))

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Disease Prediction with Chatbot", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Disease Prediction System",
        ["Diabetes", "Heart Disease", "Parkinsons"],
        icons=["activity", "heart", "person"],
        default_index=0
    )

# =========================
# Diabetes Prediction Page
# =========================
if selected == "Diabetes":
    st.title("Diabetes Prediction")
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
    with col2: Glucose = st.number_input("Glucose Level", 0, 300, 120)
    with col3: BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    with col1: SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    with col2: Insulin = st.number_input("Insulin Level", 0, 900, 80)
    with col3: BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col1: DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    with col2: Age = st.number_input("Age", 0, 120, 30)

    diab_diagnosis = ""

    if st.button("Diabetes Test Result"):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                  Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = "You have diabetes." if diab_prediction[0] == 1 else "You do not have diabetes."
        st.success(diab_diagnosis)

        # Chatbot
        if "chat_diab" not in st.session_state: st.session_state.chat_diab = []
        st.subheader("💬 Chat with Medical Assistant")
        for msg in st.session_state.chat_diab: st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask about your diabetes condition..."):
            st.session_state.chat_diab.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Diabetes", diab_diagnosis, user_query)
            st.session_state.chat_diab.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)

# =========================
# Heart Disease Prediction Page
# =========================
if selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)
    with col1: age = st.number_input("Age", 0, 120, 45)
    with col2: sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    with col3: cp = st.number_input("Chest Pain types (0-3)", 0, 3, 0)
    with col1: trestbps = st.number_input("Resting Blood Pressure", 0, 200, 120)
    with col2: chol = st.number_input("Serum Cholesterol", 0, 600, 200)
    with col3: fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
    with col1: restecg = st.number_input("Resting ECG results (0-2)", 0, 2, 1)
    with col2: thalach = st.number_input("Max Heart Rate achieved", 0, 300, 150)
    with col3: exang = st.selectbox("Exercise Induced Angina", [1, 0])
    with col1: oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
    with col2: slope = st.number_input("Slope of peak exercise ST segment", 0, 2, 1)
    with col3: ca = st.number_input("Major vessels colored by fluoroscopy", 0, 3, 0)
    with col1: thal = st.number_input("Thal (0-3)", 0, 3, 1)

    heart_diagnosis = ""

    if st.button("Heart Disease Test Result"):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,
                                                 thalach, exang, oldpeak, slope, ca, thal]])
        heart_diagnosis = "You have heart disease." if heart_prediction[0] == 1 else "You do not have heart disease."
        st.success(heart_diagnosis)

        if "chat_heart" not in st.session_state: st.session_state.chat_heart = []
        st.subheader("💬 Chat with Medical Assistant")
        for msg in st.session_state.chat_heart: st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask about your heart condition..."):
            st.session_state.chat_heart.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Heart Disease", heart_diagnosis, user_query)
            st.session_state.chat_heart.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)

# =========================
# Parkinson's Disease Prediction Page
# =========================
if selected == "Parkinsons":
    st.title("Parkinson's Disease Prediction")
    col1, col2, col3 = st.columns(3)
    with col1: fo = st.number_input("MDVP:Fo(Hz)", 50.0, 500.0, 120.0)
    with col2: fhi = st.number_input("MDVP:Fhi(Hz)", 50.0, 500.0, 150.0)
    with col3: flo = st.number_input("MDVP:Flo(Hz)", 50.0, 500.0, 85.0)
    with col1: Jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 0.02, 0.005)
    with col2: Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.01, 0.00005)
    with col3: RAP = st.number_input("MDVP:RAP", 0.0, 0.01, 0.003)
    with col1: PPQ = st.number_input("MDVP:PPQ", 0.0, 0.01, 0.003)
    with col2: DDP = st.number_input("Jitter:DDP", 0.0, 0.02, 0.01)
    with col3: Shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.05, 0.02)
    with col1: Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", 0.0, 1.0, 0.2)
    with col2: APQ3 = st.number_input("Shimmer:APQ3", 0.0, 0.05, 0.01)
    with col3: APQ5 = st.number_input("Shimmer:APQ5", 0.0, 0.05, 0.02)
    with col1: APQ = st.number_input("MDVP:APQ", 0.0, 0.05, 0.03)
    with col2: DDA = st.number_input("Shimmer:DDA", 0.0, 0.05, 0.03)
    with col3: NHR = st.number_input("NHR", 0.0, 0.05, 0.01)
    with col1: HNR = st.number_input("HNR", 0.0, 50.0, 20.0)
    with col2: RPDE = st.number_input("RPDE", 0.0, 1.0, 0.5)
    with col3: DFA = st.number_input("DFA", 0.0, 1.0, 0.6)
    with col1: spread1 = st.number_input("Spread1", -10.0, 10.0, -4.0)
    with col2: spread2 = st.number_input("Spread2", -10.0, 10.0, 0.3)
    with col3: D2 = st.number_input("D2", 0.0, 10.0, 2.0)
    with col1: PPE = st.number_input("PPE", 0.0, 1.0, 0.2)

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                                           RPDE, DFA, spread1, spread2, D2, PPE]])
        parkinsons_diagnosis = "You have Parkinson's disease." if parkinsons_prediction[0] == 1 else "You do not have Parkinson's disease."
        st.success(parkinsons_diagnosis)

        if "chat_parkinsons" not in st.session_state: st.session_state.chat_parkinsons = []
        st.subheader("💬 Chat with Medical Assistant")
        for msg in st.session_state.chat_parkinsons: st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask about your Parkinson's condition..."):
            st.session_state.chat_parkinsons.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Parkinson's disease", parkinsons_diagnosis, user_query)
            st.session_state.chat_parkinsons.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)
