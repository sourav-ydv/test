"""
Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
With OCR-based Health Report Auto-Redirect to Chatbot
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
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

    prefill = st.session_state.get("last_prediction", {})
    values = prefill.get("input", []) if prefill.get("disease") == "Diabetes" else []

    Pregnancies = st.text_input("Pregnancies", value=str(values[0]) if values else "")
    Glucose = st.text_input("Glucose Level", value=str(values[1]) if values else "")
    BloodPressure = st.text_input("Blood Pressure value", value=str(values[2]) if values else "")
    SkinThickness = st.text_input("Skin Thickness value", value=str(values[3]) if values else "")
    Insulin = st.text_input("Insulin Level", value=str(values[4]) if values else "")
    BMI = st.text_input("BMI value", value=str(values[5]) if values else "")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value", value=str(values[6]) if values else "")
    Age = st.text_input("Age", value=str(values[7]) if values else "")

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
        st.session_state['last_prediction'] = {'disease': 'Diabetes','input': user_input_d,'result': diab_status}

# ---------------------------------------------------------
# 6️⃣ Heart Disease Prediction
# ---------------------------------------------------------
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    prefill = st.session_state.get("last_prediction", {})
    values = prefill.get("input", []) if prefill.get("disease") == "Heart Disease" else []

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', value=str(values[0]) if values else "")
        sex = st.text_input('Sex (1 = Male, 0 = Female)', value=str(values[1]) if values else "")
        cp = st.text_input('Chest Pain Type (0–3)', value=str(values[2]) if values else "")
        trestbps = st.text_input('Resting Blood Pressure', value=str(values[3]) if values else "")
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)', value=str(values[4]) if values else "")
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', value=str(values[5]) if values else "")
        restecg = st.text_input('Resting ECG Results (0–2)', value=str(values[6]) if values else "")
        thalach = st.text_input('Maximum Heart Rate Achieved', value=str(values[7]) if values else "")
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)', value=str(values[8]) if values else "")
        oldpeak = st.text_input('Oldpeak', value=str(values[9]) if values else "")
        slope = st.text_input('Slope (0–2)', value=str(values[10]) if values else "")
        ca = st.text_input('Number of Major Vessels (0–3)', value=str(values[11]) if values else "")
        thal = st.text_input('Thalassemia (0,1,2)', value=str(values[12]) if values else "")

    if st.button('Heart Disease Test Result'):
        user_input_h = [int(age), int(sex), int(cp), int(trestbps), int(chol),
                        int(fbs), int(restecg), int(thalach), int(exang),
                        float(oldpeak), int(slope), int(ca), int(thal)]
        heart_prediction = heart_model.predict([user_input_h])
        if heart_prediction[0] == 1:
            st.error('The person is likely to have heart disease.')
            heart_status = 'likely to have heart disease'
        else:
            st.success('The person does not have any heart disease.')
            heart_status = 'does not have any heart disease'
        st.session_state['last_prediction'] = {'disease': 'Heart Disease','input': user_input_h,'result': heart_status}

# ---------------------------------------------------------
# 7️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
if selected == 'Parkinson’s Prediction':
    st.title("Parkinson’s Disease Prediction using ML")

    prefill = st.session_state.get("last_prediction", {})
    values = prefill.get("input", []) if prefill.get("disease") == "Parkinson’s Disease" else []

    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "D2", "DFA",
        "Spread1", "Spread2", "PPE"
    ]
    inputs = []
    for i, feature in enumerate(features):
        val = values[i] if values and i < len(values) else 0.0
        inputs.append(st.number_input(feature, value=float(val)))

    if st.button('Parkinson’s Test Result'):
        park_prediction = parkinsons_model.predict([inputs])
        if park_prediction[0] == 1:
            st.error('The person likely has Parkinson’s Disease.')
            park_status = 'likely to have Parkinson’s Disease'
        else:
            st.success('The person is healthy.')
            park_status = 'does not have Parkinson’s Disease'
        st.session_state['last_prediction'] = {'disease': 'Parkinson’s Disease','input': inputs,'result': park_status}

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

    # ✅ Reset input BEFORE widget
    if st.session_state.get("reset_chat_input", False):
        st.session_state.pop("chat_input", None)
        st.session_state["reset_chat_input"] = False

    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = ""

    # 🔄 Auto-reply from OCR general report
    if st.session_state.get("last_prediction", {}).get("disease") == "General Report":
        extracted_text = st.session_state["last_prediction"]["result"]
        if not any(msg["content"] == extracted_text for msg in st.session_state.chat_history):
            st.session_state.chat_history.append({"role": "user", "content": extracted_text})
            system_prompt = (
                "You are a helpful AI health assistant named HealthBot. "
                "Analyze the uploaded health report text. "
                "Provide general health, wellness, exercise, and diet suggestions. "
                "Do not prescribe medicine. Encourage consulting professionals."
            )
            full_prompt = f"{system_prompt}\n\nUser Health Report:\n{extracted_text}\n\nGive advice and suggestions."
            try:
                gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
                response = gemini_model.generate_content(full_prompt)
                reply = response.text
            except Exception as e:
                reply = f"⚠️ Gemini API error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Show chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='background:#1e1e1e;padding:10px;border-radius:12px;margin:8px 0;text-align:right;color:#fff;'>🧑 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#2b313e;padding:10px;border-radius:12px;margin:8px 0;text-align:left;color:#e2e2e2;'>🤖 <b>HealthBot:</b> {msg['content']}</div>", unsafe_allow_html=True)

    # Chat input
    st.text_area("💬 Type your message:", key="chat_input", height=80, placeholder="Ask about diet, fitness, or your health data...")

    col1, col2 = st.columns([4, 1])

    with col1:
        if st.button("Send", use_container_width=True):
            text = st.session_state["chat_input"].strip()
            if text:
                st.session_state.chat_history.append({"role": "user", "content": text})
                system_prompt = (
                    "You are a helpful AI health assistant named HealthBot. "
                    "Provide safe, supportive responses. "
                    "Do not prescribe medicine."
                )
                full_prompt = f"{system_prompt}\n\nUser Question: {text}"
                try:
                    gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
                    response = gemini_model.generate_content(full_prompt)
                    reply = response.text
                except Exception as e:
                    reply = f"⚠️ Gemini API error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            st.session_state["reset_chat_input"] = True
            st.rerun()

    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state["reset_chat_input"] = True
            st.rerun()

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

        # Always send extracted text to chatbot
        st.session_state['last_prediction'] = {
            'disease': "General Report",
            'input': [],
            'result': extracted_text
        }
        st.session_state["redirect_to"] = "HealthBot Assistant"
        st.rerun()
