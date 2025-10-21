"""
Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
OpenAI (primary) + Gemini fallback (gemini-2.5-flash-lite)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
import pytesseract

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
         'Parkinson’s Prediction', 'HealthBot Assistant', 'Upload Health Report'],
        icons=['activity', 'heart', 'brain', 'robot', 'file-earmark-arrow-up'],
        default_index=0
    )

# ---------------------------------------------------------
# OCR Utility
# ---------------------------------------------------------
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

# ---------------------------------------------------------
# 🔄 Redirect Handling
# ---------------------------------------------------------
if "redirect_to" in st.session_state and st.session_state["redirect_to"]:
    selected = st.session_state["redirect_to"]
    st.session_state["redirect_to"] = None

# ---------------------------------------------------------
# 5️⃣ Diabetes Prediction
# ---------------------------------------------------------
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)
    
    with col1:
    Pregnancies = st.text_input("Pregnancies")
    with col2:
    Glucose = st.text_input("Glucose Level")
    with col3:
    BloodPressure = st.text_input("Blood Pressure value")
    with col1:
    SkinThickness = st.text_input("Skin Thickness value")
    with col2:
    Insulin = st.text_input("Insulin Level")
    with col3:
    BMI = st.text_input("BMI value")
    with col1:
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2:
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
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
        cp = st.text_input('Chest Pain Type (0–3)')
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)')
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)')
        restecg = st.text_input('Resting ECG Results (0–2)')
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
        oldpeak = st.text_input('Oldpeak (ST Depression by Exercise)')
        slope = st.text_input('Slope of Peak Exercise ST Segment (0–2)')
        ca = st.text_input('Number of Major Vessels (0–3)')
        thal = st.text_input('Thalassemia (0 = Normal, 1 = Fixed, 2 = Reversible)')

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

    st.markdown("### Enter Voice Measurement Features")
    parkinsons_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "D2", "DFA",
        "Spread1", "Spread2", "PPE"
    ]

    inputs = []
    for feature in parkinsons_features:
        value = st.number_input(feature, 0.0)
        inputs.append(value)

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
# 8️⃣ HealthBot Assistant
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("⚠️ Gemini API key missing or invalid. Please check your configuration.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Auto-reply if OCR uploaded ---
    last_pred = st.session_state.get("last_prediction", None)
    if isinstance(last_pred, dict) and last_pred.get("disease") == "General Report":
        report_text = last_pred["result"]
        if not any(msg["content"] == report_text for msg in st.session_state.chat_history):
            st.session_state.chat_history.append({"role": "user", "content": report_text})
            system_prompt = (
                "You are a helpful AI health assistant named HealthBot. "
                "Analyze the uploaded health report text. "
                "Provide structured insights with: Findings, Risks, Suggestions. "
                "Do not prescribe medicine."
            )
            full_prompt = f"{system_prompt}\n\nHealth Report:\n{report_text}"
            try:
                gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
                response = gemini_model.generate_content(full_prompt)
                reply = response.text
            except Exception as e:
                reply = f"⚠️ Gemini API error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # --- Show chat history ---
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='background:#1e1e1e;padding:10px;border-radius:12px;margin:8px 0;text-align:right;color:#fff;'>🧑 <b>You:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#2b313e;padding:10px;border-radius:12px;margin:8px 0;text-align:left;color:#e2e2e2;'>🤖 <b>HealthBot:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )

    # --- Input field ---
    user_message = st.chat_input("💬 Type your message...")

    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Add last prediction context
        last_pred = st.session_state.get('last_prediction', None)
        user_context = ""
        if isinstance(last_pred, dict) and last_pred.get('disease') != "General Report":
            user_context = (
                f"\nPrevious Test Performed: {last_pred['disease']}\n"
                f"Input Values: {last_pred['input']}\n"
                f"Prediction Result: {last_pred['result']}\n"
            )

        full_prompt = (
            "You are HealthBot, a safe AI assistant.\n"
            "Always give structured and detailed answers with:\n"
            "- Findings: interpret the test values.\n"
            "- Risks: explain possible health implications.\n"
            "- Suggestions: lifestyle, diet, or follow-up actions.\n"
            "Never prescribe medicines.\n\n"
            f"{user_context}\nUser Question: {user_message}"
        )

        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(full_prompt)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Gemini API error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()   # ✅ Refresh immediately so no delay

    # --- Clear Chat button in sidebar ---
    with st.sidebar:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state['last_prediction'] = None
            st.rerun()   # ✅ Instant clear

# ---------------------------------------------------------
# 9️⃣ Upload Health Report (OCR → Chatbot only)
# ---------------------------------------------------------
if selected == "Upload Health Report":
    st.title("📑 Upload Health Report for OCR Analysis")

    uploaded_file = st.file_uploader("Upload health report image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_file)

        st.subheader("📄 Extracted Text")
        st.text(extracted_text)

        # Send extracted report text to chatbot
        st.session_state['last_prediction'] = {
            'disease': "General Report",
            'input': [],
            'result': extracted_text
        }
        st.session_state["redirect_to"] = "HealthBot Assistant"
        st.rerun()

