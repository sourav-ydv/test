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
# 3️⃣ Sidebar Menu (added Upload Health Report)
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
# Helpers: OCR + Chat generation
# ---------------------------------------------------------
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def generate_chat_reply(prompt_text: str):
    """
    Uses Gemini with settings tuned for thorough, structured answers.
    Falls back to a short message on exception.
    """
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
        response = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": 0.4,
                "top_p": 0.9,
                "max_output_tokens": 2048,   # ensure detailed outputs
            },
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

def build_system_prompt():
    # Long, structured system prompt to encourage detailed output
    return (
        "You are HealthBot, a helpful and knowledgeable AI health assistant. "
        "Your goals:\n"
        "1) Provide thorough, structured, plain-language explanations.\n"
        "2) Use headings and bullet lists: Findings, What it could mean, Lifestyle & Diet Suggestions, Red flags (see a doctor), Notes/Limitations.\n"
        "3) Reference normal ranges qualitatively (low/normal/high) without diagnosing or prescribing.\n"
        "4) Encourage consulting a qualified professional for any medical concern.\n"
        "5) Stay safe: do not prescribe medicine or make definitive diagnoses.\n"
    )

def build_prediction_context(last_pred: dict):
    """Attach detailed context about the last structured prediction if available."""
    if not last_pred or last_pred.get('disease') in (None, "General Report"):
        return ""

    disease = last_pred['disease']
    values = last_pred.get('input', [])
    if disease == "Diabetes":
        columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    elif disease == "Heart Disease":
        columns = [
            "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestECG", "MaxHR", "ExerciseAngina",
            "Oldpeak", "Slope", "CA", "Thal"
        ]
    elif disease == "Parkinson’s Disease":
        columns = [f"Feature_{i}" for i in range(1, 23)]
    else:
        columns = []

    named = "\n".join([f"- {c}: {v}" for c, v in zip(columns, values)])
    result = last_pred.get("result", "N/A")

    return (
        f"\n[Previous Prediction Context]\n"
        f"Disease/Test: {disease}\n"
        f"Inputs:\n{named}\n"
        f"Model result: {result}\n"
        f"Provide tailored, general lifestyle guidance relevant to the above, "
        f"while keeping safety constraints in mind.\n"
    )

# ---------------------------------------------------------
# 🔄 Cross-page redirect handler
# ---------------------------------------------------------
if "redirect_to" in st.session_state and st.session_state["redirect_to"]:
    selected = st.session_state["redirect_to"]
    st.session_state["redirect_to"] = None

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
        user_input_d = [
            int(Pregnancies), int(Glucose), int(BloodPressure),
            int(SkinThickness), int(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), int(Age)
        ]
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
# 8️⃣ HealthBot Assistant (detailed + upload-again)
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    # Session state init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "report_consumed" not in st.session_state:
        st.session_state.report_consumed = False
    if "show_upload_again" not in st.session_state:
        st.session_state.show_upload_again = False

    # A) If redirected from OCR upload, import once and then CONSUME it
    last_pred = st.session_state.get("last_prediction")
    if last_pred and last_pred.get("disease") == "General Report" and not st.session_state.report_consumed:
        report_text = last_pred.get("result", "").strip()
        if report_text:
            # Push the report into chat as if user said it
            st.session_state.chat_history.append({"role": "user", "content": report_text})

            system_prompt = build_system_prompt()
            full_prompt = (
                f"{system_prompt}\n\n"
                f"[Uploaded Health Report Text]\n{report_text}\n\n"
                f"Now analyze thoroughly with the requested structure."
            )
            reply = generate_chat_reply(full_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Mark as consumed so it doesn't re-post on rerun
        st.session_state.report_consumed = True
        st.session_state.show_upload_again = True  # ✅ show upload box after handling one report

    # B) Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='background:#1e1e1e;padding:10px;border-radius:12px;margin:8px 0;text-align:right;color:#fff;'>"
                f"🧑 <b>You:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#2b313e;padding:10px;border-radius:12px;margin:8px 0;text-align:left;color:#e2e2e2;'>"
                f"🤖 <b>HealthBot:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # C) Inline "Upload another report" inside chatbot after one is processed
    if st.session_state.show_upload_again:
        st.subheader("📤 Upload another health report")
        again_file = st.file_uploader(
            "Upload image (png/jpg/jpeg) — it will be analyzed and summarized here:",
            type=["png", "jpg", "jpeg"],
            key="upload_again_widget"
        )
        if again_file is not None:
            with st.spinner("Extracting text from image..."):
                extracted_text = extract_text_from_image(again_file)
            # Immediately treat it like a new user message
            st.session_state.chat_history.append({"role": "user", "content": extracted_text})

            system_prompt = build_system_prompt()
            full_prompt = (
                f"{system_prompt}\n\n"
                f"[Uploaded Health Report Text]\n{extracted_text}\n\n"
                f"Now analyze thoroughly with the requested structure."
            )
            reply = generate_chat_reply(full_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            # No need to set last_prediction for inline uploads; they live in chat only
            # Keep upload box visible for more uploads

    # D) Chat input + buttons
    def handle_send():
        text = st.session_state.chat_input.strip()
        if not text:
            return

        st.session_state.chat_history.append({"role": "user", "content": text})

        # Attach context from last structured prediction (but not General Report)
        context = build_prediction_context(st.session_state.get("last_prediction"))
        system_prompt = build_system_prompt()
        full_prompt = (
            f"{system_prompt}\n"
            f"{context}\n"
            f"[User Question]\n{text}\n"
            f"Respond comprehensively with the requested structure."
        )
        reply = generate_chat_reply(full_prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.chat_input = ""

    def clear_chat():
        # Clear everything, including uploaded report residue
        st.session_state.chat_history = []
        st.session_state.chat_input = ""
        st.session_state.show_upload_again = False
        st.session_state.report_consumed = False
        # If last_prediction is a General Report, clear it so it doesn't re-trigger
        lp = st.session_state.get("last_prediction")
        if lp and lp.get("disease") == "General Report":
            st.session_state["last_prediction"] = None

    st.text_area(
        "💬 Type your message:",
        key="chat_input",
        height=80,
        placeholder="Ask about diet, fitness, or your health data..."
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        st.button("Send", use_container_width=True, on_click=handle_send)
    with col2:
        st.button("🧹 Clear Chat", use_container_width=True, on_click=clear_chat)

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

        # Store as a 'General Report' and redirect once.
        st.session_state['last_prediction'] = {
            'disease': "General Report",
            'input': [],
            'result': extracted_text
        }
        # Reset consumption flags so chatbot processes this exactly once
        st.session_state.report_consumed = False
        st.session_state.show_upload_again = False
        st.session_state["redirect_to"] = "HealthBot Assistant"
        st.rerun()
