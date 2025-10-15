"""
Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
Now with OCR-based Health Report Analysis
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
import pytesseract
import re

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
# OCR Utility Functions
# ---------------------------------------------------------
def extract_text_from_image(uploaded_file):
    """Run OCR on uploaded image."""
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def parse_health_report(text):
    """
    Parse OCR text to extract values for known models.
    Returns:
        {"disease": <name>, "data": <list of values>}  
        OR  {"general": <text>}
    """
    text = text.lower()

    # --- Diabetes fields ---
    if "glucose" in text or "insulin" in text or "bmi" in text:
        values = {
            "Pregnancies": re.search(r"pregnancies[:\s]+(\d+)", text),
            "Glucose": re.search(r"glucose[:\s]+(\d+)", text),
            "BloodPressure": re.search(r"blood pressure[:\s]+(\d+)", text),
            "SkinThickness": re.search(r"skin thickness[:\s]+(\d+)", text),
            "Insulin": re.search(r"insulin[:\s]+(\d+)", text),
            "BMI": re.search(r"bmi[:\s]+([\d.]+)", text),
            "DiabetesPedigreeFunction": re.search(r"pedigree[:\s]+([\d.]+)", text),
            "Age": re.search(r"age[:\s]+(\d+)", text)
        }
        cleaned = [float(m.group(1)) if m else 0 for m in values.values()]
        return {"disease": "Diabetes", "data": cleaned}

    # --- Heart Disease fields ---
    if "cholesterol" in text or "blood pressure" in text or "thal" in text:
        values = {
            "Age": re.search(r"age[:\s]+(\d+)", text),
            "Sex": re.search(r"sex[:\s]+(\d+)", text),
            "ChestPain": re.search(r"chest pain[:\s]+(\d+)", text),
            "RestBP": re.search(r"blood pressure[:\s]+(\d+)", text),
            "Chol": re.search(r"cholesterol[:\s]+(\d+)", text),
            "FBS": re.search(r"fbs[:\s]+(\d+)", text),
            "RestECG": re.search(r"ecg[:\s]+(\d+)", text),
            "MaxHR": re.search(r"heart rate[:\s]+(\d+)", text),
            "ExAng": re.search(r"angina[:\s]+(\d+)", text),
            "Oldpeak": re.search(r"oldpeak[:\s]+([\d.]+)", text),
            "Slope": re.search(r"slope[:\s]+(\d+)", text),
            "CA": re.search(r"ca[:\s]+(\d+)", text),
            "Thal": re.search(r"thal[:\s]+(\d+)", text)
        }
        cleaned = [float(m.group(1)) if m else 0 for m in values.values()]
        return {"disease": "Heart Disease", "data": cleaned}

    # --- Parkinson’s fields ---
    if "jitter" in text or "shimmer" in text or "hnr" in text:
        numbers = re.findall(r"([\d.]+)", text)
        if len(numbers) >= 22:   # minimum required
            cleaned = [float(x) for x in numbers[:22]]
            return {"disease": "Parkinson’s Disease", "data": cleaned}

    # --- If general report ---
    return {"general": text}

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
        ca = st.text_input('Number of Major Vessels (0–3) Colored by Fluoroscopy')
        thal = st.text_input('Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)')

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
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#1e1e1e;padding:10px 15px;border-radius:12px;"
                    f"margin:8px 0;text-align:right;color:#fff;'>"
                    f"🧑 <b>You:</b> {msg['content']}"
                    f"</div>", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#2b313e;padding:10px 15px;border-radius:12px;"
                    f"margin:8px 0;text-align:left;color:#e2e2e2;'>"
                    f"🤖 <b>HealthBot:</b> {msg['content']}"
                    f"</div>", unsafe_allow_html=True
                )

    st.markdown("---")

    def handle_send():
        user_text = st.session_state.chat_input.strip()
        if not user_text:
            return
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        system_prompt = (
            "You are a helpful and knowledgeable AI health assistant named HealthBot. "
            "Provide general information on health, wellness, exercise, and diet. "
            "Avoid giving any medical prescriptions or diagnoses. "
            "Encourage users to consult professionals for medical concerns."
        )

        last_pred = st.session_state.get('last_prediction', None)
        user_context = ""
        if last_pred:
            disease = last_pred['disease']
            values = last_pred['input']
            input_with_names = "\n".join([str(v) for v in values])
            user_context = (
                f"\nUser recently tested for {disease}.\n"
                f"Prediction result: {last_pred['result']}\n"
                f"Values: {input_with_names}\n"
                "Give lifestyle advice, diet tips, or precautions related to this data."
            )

        full_prompt = f"{system_prompt}\n{user_context}\n\nUser Question: {user_text}"

        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(full_prompt)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Gemini API error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.chat_input = ""

    def clear_chat():
        st.session_state.chat_history = []
        st.session_state.chat_input = ""

    st.text_area("💬 Type your message:", key="chat_input", height=80,
                 placeholder="Ask about diet, fitness, or your health data...")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.button("Send", use_container_width=True, on_click=handle_send)
    with col2:
        st.button("🧹 Clear Chat", use_container_width=True, on_click=clear_chat)

# ---------------------------------------------------------
# 9️⃣ Upload Health Report (OCR Integration)
# ---------------------------------------------------------
if selected == "Upload Health Report":
    st.title("📑 Upload Health Report for OCR Analysis")

    uploaded_file = st.file_uploader("Upload health report image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_file)

        st.subheader("📄 Extracted Text")
        st.text(extracted_text)

        parsed = parse_health_report(extracted_text)

        if "disease" in parsed:
            st.success(f"✅ Data related to {parsed['disease']} found in report.")
            st.session_state['last_prediction'] = {
                'disease': parsed['disease'],
                'input': parsed['data'],
                'result': "Pending prediction"
            }
            st.info("This data has been auto-filled. Please go to the respective prediction page to run the model.")

        else:
    st.warning("ℹ️ No direct match with supported diseases. Opening HealthBot for advice...")
    st.session_state['last_prediction'] = {
        'disease': "General Report",
        'input': [],
        'result': parsed['general']
    }
    st.session_state["go_to_healthbot"] = True

