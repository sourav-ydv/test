"""
Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
With OCR-based Health Report Auto-Redirect + Auto-Fill + Auto-Reply
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
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
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def parse_health_report(text):
    text = text.lower()

    # Diabetes
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

    # Heart
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

    # Parkinson’s
    if "jitter" in text or "shimmer" in text or "hnr" in text:
        numbers = re.findall(r"([\d.]+)", text)
        if len(numbers) >= 22:
            cleaned = [float(x) for x in numbers[:22]]
            return {"disease": "Parkinson’s Disease", "data": cleaned}

    return {"general": text}

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
