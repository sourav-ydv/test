# -*- coding: utf-8 -*-
"""
Multi-Disease Prediction System + Smart Gemini HealthBot
Author: Sourav Yadav
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
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
        '🧬 Disease Prediction System',
        ['🏠 Home', '💉 Diabetes Prediction', '❤️ Heart Disease Prediction',
         '🧠 Parkinson’s Prediction', '🤖 HealthBot Assistant'],
        icons=['house', 'activity', 'heart', 'brain', 'robot'],
        default_index=0
    )

# ---------------------------------------------------------
# 4️⃣ Home Page
# ---------------------------------------------------------
if selected == '🏠 Home':
    st.title("🏥 Multi-Disease Prediction & Health Assistant")
    st.write("""
    Welcome to the **AI-Powered Health Prediction System**!  
    This app can:
    - Predict your risk for **Diabetes**, **Heart Disease**, and **Parkinson’s Disease**  
    - Chat with a built-in **Health Assistant** powered by **Gemini AI**  

    ⚠️ *Disclaimer:* This app is for educational and informational purposes only — not medical advice.
    """)

# ---------------------------------------------------------
# 5️⃣ Diabetes Prediction
# ---------------------------------------------------------
if selected == '💉 Diabetes Prediction':
    st.title("💉 Diabetes Prediction using ML")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0)
        Glucose = st.number_input("Glucose Level", 0)
        BloodPressure = st.number_input("Blood Pressure", 0)
        SkinThickness = st.number_input("Skin Thickness", 0)
    with col2:
        Insulin = st.number_input("Insulin Level", 0)
        BMI = st.number_input("BMI", 0.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0)
        Age = st.number_input("Age", 0)

    if st.button('🔍 Diabetes Test Result'):
        user_input_d = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "Blood Pressure": BloodPressure,
            "Skin Thickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age
        }

        diab_prediction = diabetes_model.predict([list(user_input_d.values())])
        if diab_prediction[0] == 1:
            st.error('⚠️ The person is likely to have diabetes.')
            diab_status = 'likely to have diabetes'
        else:
            st.success('✅ The person is not diabetic.')
            diab_status = 'not diabetic'

        st.session_state.last_prediction = {
            "disease": "Diabetes",
            "input": user_input_d,
            "result": diab_status
        }

# ---------------------------------------------------------
# 6️⃣ Heart Disease Prediction
# ---------------------------------------------------------
if selected == '❤️ Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 0)
        sex = st.selectbox('Sex (1=Male, 0=Female)', [0, 1])
        cp = st.number_input('Chest Pain Type', 0)
        trestbps = st.number_input('Resting Blood Pressure', 0)
    with col2:
        chol = st.number_input('Cholesterol (mg/dl)', 0)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
        restecg = st.number_input('Rest ECG Result', 0)
        thalach = st.number_input('Max Heart Rate', 0)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', [0, 1])
        oldpeak = st.number_input('ST Depression', 0.0)
        slope = st.number_input('Slope of ST Segment', 0)
        ca = st.number_input('No. of Major Vessels', 0)
        thal = st.number_input('Thal (0=Normal,1=Fixed Defect,2=Reversible Defect)', 0)

    if st.button('🔍 Heart Disease Test Result'):
        user_input_h = {
            "Age": age, "Sex": sex, "Chest Pain Type": cp, "Resting BP": trestbps,
            "Cholesterol": chol, "Fasting Sugar": fbs, "Rest ECG": restecg,
            "Max Heart Rate": thalach, "Exercise Angina": exang, "Oldpeak": oldpeak,
            "Slope": slope, "Major Vessels": ca, "Thal": thal
        }

        heart_prediction = heart_model.predict([list(user_input_h.values())])
        if heart_prediction[0] == 1:
            st.error('💔 The person is likely to have heart disease.')
            heart_status = 'likely to have heart disease'
        else:
            st.success('❤️ The person does not have heart disease.')
            heart_status = 'does not have heart disease'

        st.session_state.last_prediction = {
            "disease": "Heart Disease",
            "input": user_input_h,
            "result": heart_status
        }

# ---------------------------------------------------------
# 7️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
if selected == '🧠 Parkinson’s Prediction':
    st.title("🧠 Parkinson’s Disease Prediction using ML")

    inputs = {}
    for i in range(1, 23):
        inputs[f'Feature {i}'] = st.number_input(f'Feature {i}', 0.0)

    if st.button('🔍 Parkinson’s Test Result'):
        park_prediction = parkinsons_model.predict([list(inputs.values())])
        if park_prediction[0] == 1:
            st.error('⚠️ The person likely has Parkinson’s Disease.')
            park_status = 'likely to have Parkinson’s Disease'
        else:
            st.success('✅ The person is healthy.')
            park_status = 'does not have Parkinson’s Disease'

        st.session_state.last_prediction = {
            "disease": "Parkinson’s Disease",
            "input": inputs,
            "result": park_status
        }

# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant (Gemini-only)
# ---------------------------------------------------------
if selected == '🤖 HealthBot Assistant':
    # 🎨 Styling
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #e8e8e8;
        }
        .chat-container {
            background-color: #0e1117;
            padding: 15px;
            border-radius: 12px;
            min-height: 400px;
        }
        .user-msg {
            background-color: #1f2937;
            color: #ffffff;
            border-radius: 18px;
            padding: 10px 15px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
            align-self: flex-end;
        }
        .bot-msg {
            background-color: #1e3a5f;
            color: #e0f7fa;
            border-radius: 18px;
            padding: 10px 15px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
            align-self: flex-start;
        }
        textarea {
            background-color: #1f2937 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #374151 !important;
        }
        div.stButton > button {
            background-color: #007acc;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 8px 16px;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #005f99;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🤖 AI HealthBot Assistant")

    # Configure Gemini
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    except Exception as e:
        st.error("⚠️ Gemini API key missing or invalid. Cannot start chatbot.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'><b>🧑 You:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-msg'><b>🤖 HealthBot:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: gray; text-align:center;'>Start chatting below 👇</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input
    st.markdown("---")
    user_input = st.text_area("💬 Type your message:", key="chat_input", height=80,
                              placeholder="Ask about your prediction results or general health advice...")
    send_btn = st.button("Send", use_container_width=True)

    # Send
    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        system_prompt = (
            "You are HealthBot, a friendly and knowledgeable AI health assistant. "
            "You provide general wellness and prevention advice based on prediction data. "
            "Do not provide diagnosis or treatment. If something sounds serious, advise seeing a doctor."
        )

        context = ""
        last_pred = st.session_state.get("last_prediction", None)
        if last_pred:
            disease = last_pred.get("disease", "Unknown")
            result = last_pred.get("result", "Unknown")
            inputs = last_pred.get("input", {})
            formatted_inputs = "\n".join([f"- {k}: {v}" for k, v in inputs.items()])
            context = (
                f"\n\n📊 **Prediction Context:**\n"
                f"Disease: {disease}\n"
                f"Result: {result}\n"
                f"User Input Data:\n{formatted_inputs}\n"
            )

        prompt = f"{system_prompt}{context}\n\nUser Question: {user_input}"

        try:
            response = gemini_model.generate_content(prompt)
            reply = response.text.strip()
        except Exception as e:
            reply = f"⚠️ Gemini API error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

# ---------------------------------------------------------
# 9️⃣ Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("⚕️ Powered by Gemini AI & ML Models — Not a substitute for professional medical advice.")
