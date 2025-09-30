import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from transformers import pipeline

# =========================
# Hugging Face Chatbot Setup
# =========================
@st.cache_resource
def load_chatbot():
    return pipeline("text2text-generation", model="google/flan-t5-small")

chatbot = load_chatbot()

def get_chatbot_response(disease_name, diagnosis, user_query, summary=False):
    """
    Generate response using Hugging Face Flan-T5.
    If summary=True, give concise summary of the diagnosis and advice.
    Strictly answer only disease-related queries.
    """
    if not summary:
        # Disease-only filter
        if not any(word in user_query.lower() for word in [disease_name.lower(), "symptom", "treatment", "diet", "exercise", "doctor", "medicine", "condition", "health", "cure", "prevent", "risk"]):
            return f"I can only answer questions about {disease_name}. Please ask about symptoms, treatments, lifestyle, or precautions."

        prompt = (
            f"You are a helpful medical assistant. The user has {disease_name}. "
            f"Diagnosis: {diagnosis}. "
            f"The user asked: {user_query}\n\n"
            f"Provide a clear, helpful answer including suggestions, lifestyle tips, and a disclaimer to consult a doctor."
        )
    else:
        # Summary prompt
        prompt = (
            f"You are a helpful medical assistant. The user has {disease_name}. "
            f"Diagnosis: {diagnosis}. "
            f"Provide a concise summary of the condition, including severity, lifestyle suggestions, precautions, and any advice. "
            f"Make it short, clear, and include a disclaimer to consult a doctor."
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
# Diabetes Prediction
# =========================
if selected == "Diabetes":
    st.title("Diabetes Prediction")
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    diab_diagnosis = ""

    if st.button("Diabetes Test Result"):
        try:
            user_input = [int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness),
                          int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = "You have diabetes." if diab_prediction[0] == 1 else "You do not have diabetes."
        except:
            diab_diagnosis = "⚠️ Invalid input, please enter numeric values."
        st.success(diab_diagnosis)

        # Chatbot Section
        if "chat_diab" not in st.session_state: st.session_state.chat_diab = []
        st.subheader("💬 Chat with Diabetes Assistant")
        for msg in st.session_state.chat_diab: st.chat_message(msg["role"]).write(msg["content"])

        if st.button("📋 Get Summary of Condition"):
            summary_text = get_chatbot_response("Diabetes", diab_diagnosis, "", summary=True)
            st.info(summary_text)

        if user_query := st.chat_input("Ask about your diabetes condition..."):
            st.session_state.chat_diab.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Diabetes", diab_diagnosis, user_query)
            st.session_state.chat_diab.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)


# =========================
# Heart Disease Prediction
# =========================
if selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    age = st.text_input("Age")
    sex = st.text_input("Sex (0 = Female, 1 = Male)")
    cp = st.text_input("Chest Pain Type (0–3)")
    trestbps = st.text_input("Resting Blood Pressure")
    chol = st.text_input("Cholesterol")
    fbs = st.text_input("Fasting Blood Sugar >120 mg/dl (0/1)")
    restecg = st.text_input("Resting ECG (0–2)")
    thalach = st.text_input("Max Heart Rate")
    exang = st.text_input("Exercise Induced Angina (0/1)")
    oldpeak = st.text_input("ST Depression")
    slope = st.text_input("Slope of Peak Exercise ST (0–2)")
    ca = st.text_input("Major vessels colored (0–3)")
    thal = st.text_input("Thal (0–3)")

    heart_diagnosis = ""

    if st.button("Heart Disease Test Result"):
        try:
            user_input = [int(age), int(sex), int(cp), int(trestbps), int(chol),
                          int(fbs), int(restecg), int(thalach), int(exang),
                          float(oldpeak), int(slope), int(ca), int(thal)]
            heart_prediction = heart_model.predict([user_input])
            heart_diagnosis = "You have heart disease." if heart_prediction[0] == 1 else "You do not have heart disease."
        except:
            heart_diagnosis = "⚠️ Invalid input, please enter numeric values."
        st.success(heart_diagnosis)

        if "chat_heart" not in st.session_state: st.session_state.chat_heart = []
        st.subheader("💬 Chat with Heart Disease Assistant")
        for msg in st.session_state.chat_heart: st.chat_message(msg["role"]).write(msg["content"])

        if st.button("📋 Get Summary of Condition"):
            summary_text = get_chatbot_response("Heart Disease", heart_diagnosis, "", summary=True)
            st.info(summary_text)

        if user_query := st.chat_input("Ask about your heart condition..."):
            st.session_state.chat_heart.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Heart Disease", heart_diagnosis, user_query)
            st.session_state.chat_heart.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)


# =========================
# Parkinson's Disease Prediction
# =========================
if selected == "Parkinsons":
    st.title("Parkinson's Disease Prediction")
    fo = st.text_input("MDVP:Fo(Hz)")
    fhi = st.text_input("MDVP:Fhi(Hz)")
    flo = st.text_input("MDVP:Flo(Hz)")
    Jitter_percent = st.text_input("MDVP:Jitter(%)")
    Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
    RAP = st.text_input("MDVP:RAP")
    PPQ = st.text_input("MDVP:PPQ")
    DDP = st.text_input("Jitter:DDP")
    Shimmer = st.text_input("MDVP:Shimmer")
    Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
    APQ3 = st.text_input("Shimmer:APQ3")
    APQ5 = st.text_input("Shimmer:APQ5")
    APQ = st.text_input("MDVP:APQ")
    DDA = st.text_input("Shimmer:DDA")
    NHR = st.text_input("NHR")
    HNR = st.text_input("HNR")
    RPDE = st.text_input("RPDE")
    DFA = st.text_input("DFA")
    spread1 = st.text_input("Spread1")
    spread2 = st.text_input("Spread2")
    D2 = st.text_input("D2")
    PPE = st.text_input("PPE")

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = "You have Parkinson's disease." if parkinsons_prediction[0] == 1 else "You do not have Parkinson's disease."
        except:
            parkinsons_diagnosis = "⚠️ Invalid input, please enter numeric values."
        st.success(parkinsons_diagnosis)

        if "chat_parkinsons" not in st.session_state: st.session_state.chat_parkinsons = []
        st.subheader("💬 Chat with Parkinson's Assistant")
        for msg in st.session_state.chat_parkinsons: st.chat_message(msg["role"]).write(msg["content"])

        if st.button("📋 Get Summary of Condition"):
            summary_text = get_chatbot_response("Parkinson's disease", parkinsons_diagnosis, "", summary=True)
            st.info(summary_text)

        if user_query := st.chat_input("Ask about your Parkinson's condition..."):
            st.session_state.chat_parkinsons.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            bot_reply = get_chatbot_response("Parkinson's disease", parkinsons_diagnosis, user_query)
            st.session_state.chat_parkinsons.append({"role": "assistant", "content": bot_reply})
            st.chat_message("assistant").write(bot_reply)
