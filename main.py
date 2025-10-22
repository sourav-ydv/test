import streamlit as st
import pickle
import sqlite3, hashlib
from streamlit_option_menu import option_menu
import google.generativeai as genai
from PIL import Image
import pytesseract
import ast

# ---------------------------------------------------------
# 0️⃣ Database Setup
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect("healthapp.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    disease TEXT,
                    input_values TEXT,
                    result TEXT,
                    chat_history TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect("healthapp.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                  (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = sqlite3.connect("healthapp.db")
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[1] == hash_password(password):
        return user[0]
    return None

def save_history(user_id, disease, input_values, result, chat_history):
    conn = sqlite3.connect("healthapp.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, disease, input_values, result, chat_history) VALUES (?,?,?,?,?)",
              (user_id, disease, str(input_values), result, str(chat_history)))
    conn.commit()
    conn.close()

def load_history(user_id):
    conn = sqlite3.connect("healthapp.db")
    c = conn.cursor()
    c.execute("SELECT disease, input_values, result, chat_history, timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 5", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

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
# 🔑 Authentication
# ---------------------------------------------------------
if "user_id" not in st.session_state:
    st.title("🔑 Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_username = st.text_input("Username (Login)", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            user_id = login_user(login_username, login_password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["username"] = login_username
                st.success(f"Welcome back, {login_username}! 🎉")
                past = load_history(user_id)
                if past:
                    st.session_state["chat_history"] = ast.literal_eval(past[0][3])
                else:
                    st.session_state["chat_history"] = []
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        reg_username = st.text_input("Username (Register)", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register", key="reg_btn"):
            if create_user(reg_username, reg_password):
                st.success("✅ Account created! Please login now.")
            else:
                st.error("⚠️ Username already exists. Try another.")

    st.stop()

# ---------------------------------------------------------
# Sidebar Menu
# ---------------------------------------------------------
with st.sidebar:
    st.success(f"👤 Logged in as {st.session_state['username']}")
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state.clear()
        st.rerun()

    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction',
         'Parkinson’s Prediction', 'HealthBot Assistant', 'Upload Health Report'],
        icons=['activity', 'heart', 'brain', 'robot', 'file-earmark-arrow-up'],
        default_index=0
    )

    st.subheader("📜 Past Predictions & Chats")
    history = load_history(st.session_state["user_id"])
    if history:
        for i, h in enumerate(history):
            with st.expander(f"{h[0]} → {h[2]} ({h[4]})", expanded=False):
                st.write("**Input Values:**", h[1])
                st.write("**Result:**", h[2])
                
                # Show saved chat history
                try:
                    chat_hist = ast.literal_eval(h[3])
                    if isinstance(chat_hist, list) and len(chat_hist) > 0:
                        st.markdown("**Chat History:**")
                        for msg in chat_hist:
                            role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
                            st.markdown(f"- **{role}:** {msg['content']}")
                    else:
                        st.info("No chat history saved for this record.")
                except Exception:
                    st.warning("⚠️ Could not load chat history.")
                
                # Restore button
                if st.button(f"🔄 Restore Session {i+1}", key=f"restore_{i}"):
                    st.session_state["chat_history"] = ast.literal_eval(h[3]) if h[3] else []
                    st.session_state["last_prediction"] = {
                        "disease": h[0],
                        "input": h[1],
                        "result": h[2]
                    }
                    st.session_state["redirect_to"] = "HealthBot Assistant"
                    st.rerun()


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
        Pregnancies = st.text_input("Pregnancies", key="d_preg")
    with col2:
        Glucose = st.text_input("Glucose Level", key="d_glu")
    with col3:
        BloodPressure = st.text_input("Blood Pressure", key="d_bp")
    with col1:
        SkinThickness = st.text_input("Skin Thickness", key="d_skin")
    with col2:
        Insulin = st.text_input("Insulin Level", key="d_insulin")
    with col3:
        BMI = st.text_input("BMI", key="d_bmi")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DPF", key="d_dpf")
    with col2:
        Age = st.text_input("Age", key="d_age")

    if st.button('Diabetes Test Result', key="d_btn"):
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
        save_history(st.session_state["user_id"], "Diabetes", user_input_d, diab_status, st.session_state.get("chat_history", []))

# ---------------------------------------------------------
# 6️⃣ Heart Disease Prediction
# ---------------------------------------------------------
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', key="h_age")
        sex = st.text_input('Sex (1=Male,0=Female)', key="h_sex")
        cp = st.text_input('Chest Pain Type (0–3)', key="h_cp")
        trestbps = st.text_input('Resting BP', key="h_bp")
        chol = st.text_input('Cholesterol', key="h_chol")
    with col2:
        fbs = st.text_input('Fasting Blood Sugar (1/0)', key="h_fbs")
        restecg = st.text_input('Resting ECG (0–2)', key="h_ecg")
        thalach = st.text_input('Max Heart Rate', key="h_hr")
        exang = st.text_input('Exercise Angina (1/0)', key="h_exang")
    with col3:
        oldpeak = st.text_input('Oldpeak', key="h_oldpeak")
        slope = st.text_input('Slope (0–2)', key="h_slope")
        ca = st.text_input('Major Vessels (0–3)', key="h_ca")
        thal = st.text_input('Thalassemia (0–2)', key="h_thal")

    if st.button('Heart Disease Test Result', key="h_btn"):
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
        save_history(st.session_state["user_id"], "Heart Disease", user_input_h, heart_status, st.session_state.get("chat_history", []))

# ---------------------------------------------------------
# 7️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
if selected == "Parkinson’s Prediction":
    st.title("Parkinson’s Disease Prediction using ML")

    col1, col2, col3, col4 = st.columns(4)  
    
    with col1:
        fo = st.text_input('Fo (Hz)', key="p_fo")
        Jitter_Abs = st.text_input('Jitter (Abs)', key="p_jabs")
        Shimmer = st.text_input('Shimmer', key="p_shimmer")
        APQ = st.text_input('APQ', key="p_apq")
        RPDE = st.text_input('RPDE', key="p_rpde")
        D2 = st.text_input('D2', key="p_d2")
    
    with col2:
        fhi = st.text_input('Fhi (Hz)', key="p_fhi")
        RAP = st.text_input('RAP', key="p_rap")
        Shimmer_dB = st.text_input('Shimmer (dB)', key="p_shdb")
        DDA = st.text_input('DDA', key="p_dda")
        DFA = st.text_input('DFA', key="p_dfa")
        PPE = st.text_input('PPE', key="p_ppe")
    
    with col3:
        flo = st.text_input('Flo (Hz)', key="p_flo")
        PPQ = st.text_input('PPQ', key="p_ppq")
        APQ3 = st.text_input('APQ3', key="p_apq3")
        NHR = st.text_input('NHR', key="p_nhr")
        spread1 = st.text_input('Spread1', key="p_sp1")

    with col4:
        Jitter_percent = st.text_input('Jitter (%)', key="p_jper")
        DDP = st.text_input('DDP', key="p_ddp")
        APQ5 = st.text_input('APQ5', key="p_apq5")
        HNR = st.text_input('HNR', key="p_hnr")
        spread2 = st.text_input('Spread2', key="p_sp2")
    
    if st.button("Parkinson's Test Result", key="p_btn"):
        try:
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent),
                float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            if parkinsons_prediction[0] == 1:
                st.error("The person likely has Parkinson’s Disease.")
                park_status = "likely to have Parkinson’s Disease"
            else:
                st.success("The person is healthy.")
                park_status = "does not have Parkinson’s Disease"

            st.session_state['last_prediction'] = {
                'disease': "Parkinson’s Disease",
                'input': user_input,
                'result': park_status
            }
            save_history(st.session_state["user_id"], "Parkinson’s Disease", user_input, park_status, st.session_state.get("chat_history", []))
        except ValueError:
            st.error("⚠️ Please fill all fields with valid numeric values.")

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
# 9️⃣ Upload Health Report
# ---------------------------------------------------------
if selected == "Upload Health Report":
    st.title("📑 Upload Health Report for OCR Analysis")
    uploaded_file = st.file_uploader("Upload health report image", type=["png", "jpg", "jpeg"], key="ocr_upload")
    if uploaded_file is not None:
        extracted_text = extract_text_from_image(uploaded_file)
        st.subheader("📄 Extracted Text")
        st.text(extracted_text)
        st.session_state['last_prediction'] = {
            'disease': "General Report",
            'input': [],
            'result': extracted_text
        }
        st.session_state["redirect_to"] = "HealthBot Assistant"
        st.rerun()


