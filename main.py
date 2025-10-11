# -*- coding: utf-8 -*-
"""
Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
OpenAI (primary) + Gemini fallback (gemini-2.5-flash-lite)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
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
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction',
         'Parkinson’s Prediction', 'HealthBot Assistant'],
        icons=['activity', 'heart', 'brain', 'robot'],
        default_index=0
    )

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
        sex = st.text_input('Sex (1=Male, 0=Female)')
        cp = st.text_input('Chest Pain types')
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)')
        restecg = st.text_input('Resting Electrocardiographic results')
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=True, 0=False)')
        oldpeak = st.text_input('ST depression induced by exercise')
        slope = st.text_input('Slope of the peak exercise ST segment')
        ca = st.text_input('Major vessels colored by fluoroscopy')
        thal = st.text_input('thal (0=Normal, 1=Fixed defect, 2=Reversable defect)')

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

    inputs = []
    for i in range(1, 23):
        inputs.append(st.number_input(f'Feature {i}', 0.0))

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
# 8️⃣ HealthBot Assistant (ChatGPT-like UI)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant (ChatGPT-like UI with Context)
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("AI HealthBot Assistant")

    # --- API Initialization ---
    use_openai = False
    client = None
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        use_openai = True
    except Exception:
        st.warning("OpenAI key missing or invalid. Will use Gemini fallback.")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        use_gemini = True
    except Exception:
        use_gemini = False
        if not use_openai:
            st.error("No OpenAI or Gemini API key found. Cannot generate replies.")
            st.stop()

    # --- Chat Memory ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # --- Display Chat Messages ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='background-color:#1e1e1e;padding:10px 15px;border-radius:12px;
                margin:8px 0;text-align:right;color:#fff;'>
                🧑 <b>You:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#2b313e;padding:10px 15px;border-radius:12px;
                margin:8px 0;text-align:left;color:#e2e2e2;'>
                🤖 <b>HealthBot:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)

    # --- Input Box ---
    # --- Input Box ---
    st.markdown("---")
    
    # Create a temporary variable to store user input safely
    user_input = st.text_area(
        "💬 Type your message:",
        key="chat_input",
        height=80,
        placeholder="Enter your question here..."
    )
    send_btn = st.button("Send", use_container_width=True)
    
    # --- When user sends message ---
    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    
        # SYSTEM PROMPT
        system_prompt = (
            "You are a professional, friendly AI health assistant. "
            "Provide general health guidance and wellness information. "
            "Focus on lifestyle, diet, exercise, and safety precautions. "
            "Never give prescriptions or medical diagnoses. "
            "If something sounds serious, advise seeing a doctor."
        )
    
        # --- Add latest prediction context (column names + values) ---
        last_pred = st.session_state.get('last_prediction', None)
        user_context = ""
        if last_pred:
            disease = last_pred['disease']
            values = last_pred['input']
    
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
    
            input_with_names = "\n".join(
                [f"{c}: {v}" for c, v in zip(columns, values)]
            )
    
            user_context = (
                f"\n\nUser recently tested for {disease}.\n"
                f"Input details:\n{input_with_names}\n"
                f"Prediction result: {last_pred['result']}\n"
                "Provide advice or lifestyle recommendations based on this data."
            )
    
        full_prompt = f"{system_prompt}\n{user_context}\n\nUser Question: {user_input}"
        reply = ""
    
        # --- Try OpenAI ---
        if use_openai:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=400,
                    temperature=0.7,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                if "insufficient_quota" in str(e) or "429" in str(e):
                    use_openai = False
                else:
                    reply = f"Error generating reply: {e}"
    
        # --- Gemini Fallback ---
        if not use_openai and use_gemini:
            try:
                gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
                gemini_response = gemini_model.generate_content(full_prompt)
                reply = gemini_response.text
            except Exception as ge:
                reply = f"Gemini API error: {ge}"
    
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    
        # 🧹 Reset chat input properly before rerun
        st.session_state["chat_input"] = ""
        st.experimental_rerun()

