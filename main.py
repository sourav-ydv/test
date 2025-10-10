# -*- coding: utf-8 -*-
"""
Multi-Disease Prediction System + Smart HealthBot
✅ OpenAI (primary) + Gemini fallback (free)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
import google.generativeai as genai

# ---------------------------------------------------------
# 1️⃣ Load Models
# ---------------------------------------------------------
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# ---------------------------------------------------------
# 2️⃣ Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")

# ---------------------------------------------------------
# 3️⃣ Sidebar
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
# 4️⃣ Home
# ---------------------------------------------------------
if selected == '🏠 Home':
    st.title("🏥 Multi-Disease Prediction & Health Assistant")
    st.write("""
    Welcome to the **AI-Powered Health Prediction System**!  
    This app can:
    - Predict your risk for **Diabetes**, **Heart Disease**, and **Parkinson’s Disease**  
    - Chat with a built-in **Health Assistant** that provides general lifestyle and wellness advice  
      
    ⚠️ *Disclaimer:* This app is for educational and informational purposes only — not medical advice.
    """)

# ---------------------------------------------------------
# 5️⃣ Diabetes Prediction
# ---------------------------------------------------------
if selected == '💉 Diabetes Prediction':
    st.title("💉 Diabetes Prediction using ML")

    Pregnancies = st.number_input("Pregnancies", 0)
    Glucose = st.number_input("Glucose Level", 0)
    BloodPressure = st.number_input("Blood Pressure value", 0)
    SkinThickness = st.number_input("Skin Thickness value", 0)
    Insulin = st.number_input("Insulin Level", 0)
    BMI = st.number_input("BMI value", 0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value", 0.0)
    Age = st.number_input("Age", 0)

    if st.button('🔍 Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
                                                   SkinThickness, Insulin, BMI,
                                                   DiabetesPedigreeFunction, Age]])
        if diab_prediction[0] == 1:
            st.error('⚠️ The person is likely to have diabetes.')
        else:
            st.success('✅ The person is not diabetic.')

# ---------------------------------------------------------
# 6️⃣ Heart Disease Prediction
# ---------------------------------------------------------
if selected == '❤️ Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction using ML")

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

    if st.button('🔍 Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,
                                                 thalach, exang, oldpeak, slope, ca, thal]])
        if heart_prediction[0] == 1:
            st.error('💔 The person is likely to have heart disease.')
        else:
            st.success('❤️ The person does not have any heart disease.')

# ---------------------------------------------------------
# 7️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
if selected == '🧠 Parkinson’s Prediction':
    st.title("🧠 Parkinson’s Disease Prediction using ML")

    inputs = []
    for i in range(1, 23):
        inputs.append(st.number_input(f'Feature {i}', 0.0))

    if st.button('🔍 Parkinson’s Test Result'):
        park_prediction = parkinsons_model.predict([inputs])
        if park_prediction[0] == 1:
            st.error('⚠️ The person likely has Parkinson’s Disease.')
        else:
            st.success('✅ The person is healthy.')

# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant (OpenAI + Gemini fallback)
# ---------------------------------------------------------
if selected == '🤖 HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    # Try OpenAI first
    client = None
    use_openai = False

    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        use_openai = True
    except Exception:
        st.warning("⚠️ OpenAI key missing or invalid. Using Gemini instead.")
        use_openai = False

    # Gemini fallback config
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    elif not use_openai:
        st.error("⚠️ No OpenAI or Gemini API key available. Add at least one in Streamlit secrets.")
        st.stop()

    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area("💬 Ask about health, diet, or exercise:", placeholder="e.g., What foods help manage blood sugar?")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            system_prompt = (
                "You are a professional, friendly AI health assistant. "
                "Provide general health guidance and wellness information. "
                "Focus on lifestyle, diet, exercise, and safety precautions. "
                "Never give prescriptions or medical diagnoses. "
                "If something sounds serious, advise seeing a doctor."
            )

            try:
                if use_openai:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *st.session_state.chat_history,
                        ],
                        max_tokens=300,
                        temperature=0.7,
                    )
                    reply = response.choices[0].message.content
                else:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    full_prompt = system_prompt + "\n\nUser: " + user_input
                    gemini_response = model.generate_content(full_prompt)
                    reply = gemini_response.text

                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            except Exception as e:
                reply = f"⚠️ Error generating reply: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Display chat
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 HealthBot:** {msg['content']}")

# ---------------------------------------------------------
# 9️⃣ Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("⚕️ Powered by OpenAI GPT / Gemini AI & ML models — Not a substitute for professional medical advice.")
