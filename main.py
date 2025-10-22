# main.py
# Multi-Disease Prediction System + Smart HealthBot (ChatGPT-style)
# Login + per-user persistent history (predictions + chats)
# SQLite storage, OCR, Gemini API (via st.secrets["GEMINI_API_KEY"])

import streamlit as st
import pickle
import sqlite3
import hashlib
import json
from datetime import datetime
from streamlit_option_menu import option_menu
import google.generativeai as genai
from PIL import Image
import pytesseract

# =========================
# 0) DB — Tables & Helpers
# =========================
DB_PATH = "healthapp.db"

def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn(); cur = con.cursor()
    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL
    )""")
    # Predictions (separate from chats)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      disease TEXT NOT NULL,
      input_values TEXT NOT NULL,      -- JSON
      result TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    # Chats: each row = one session (list of messages as JSON)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      messages TEXT NOT NULL,          -- JSON: [{"role":"user/assistant","content": "..."}]
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    con.commit(); con.close()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def create_user(username: str, password: str) -> bool:
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("INSERT INTO users(username,password_hash) VALUES(?,?)",
                    (username, hash_password(password)))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()

def login_user(username: str, password: str):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    con.close()
    if row and row[1] == hash_password(password):
        return row[0]
    return None

def save_prediction(user_id: int, disease: str, inputs: list, result: str):
    con = db_conn(); cur = con.cursor()
    cur.execute(
        "INSERT INTO predictions(user_id,disease,input_values,result,timestamp) VALUES(?,?,?,?,?)",
        (user_id, disease, json.dumps(inputs), result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    con.commit(); con.close()

def load_predictions(user_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("""
        SELECT disease, input_values, result, timestamp
        FROM predictions
        WHERE user_id=?
        ORDER BY timestamp DESC
    """, (user_id,))
    rows = cur.fetchall()
    con.close()
    # Return parsed JSON for inputs
    return [(d, json.loads(inp), r, ts) for (d, inp, r, ts) in rows]

def create_chat_session(user_id: int) -> int:
    con = db_conn(); cur = con.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO chats(user_id,messages,created_at,updated_at) VALUES(?,?,?,?)",
        (user_id, json.dumps([]), now, now)
    )
    chat_id = cur.lastrowid
    con.commit(); con.close()
    return chat_id

def load_chat_sessions(user_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("""
        SELECT id, messages, created_at, updated_at
        FROM chats
        WHERE user_id=?
        ORDER BY updated_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    con.close()
    # Parse messages JSON
    return [(cid, json.loads(msgs), c_at, u_at) for (cid, msgs, c_at, u_at) in rows]

def load_chat_by_id(chat_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT messages FROM chats WHERE id=?", (chat_id,))
    row = cur.fetchone()
    con.close()
    return json.loads(row[0]) if row else []

def save_chat_messages(chat_id: int, messages: list):
    con = db_conn(); cur = con.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("UPDATE chats SET messages=?, updated_at=? WHERE id=?",
                (json.dumps(messages), now, chat_id))
    con.commit(); con.close()

def delete_chat(chat_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    con.commit(); con.close()

# =========================
# 1) Models + Page Config
# =========================
init_db()

# Load models (keep your existing .sav filenames)
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")

# =========================
# 2) Auth (Login/Register)
# =========================
if "user_id" not in st.session_state:
    st.title("🔑 Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_username = st.text_input("Username (Login)", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            uid = login_user(login_username, login_password)
            if uid:
                st.session_state.user_id = uid
                st.session_state.username = login_username
                st.success(f"Welcome back, {login_username}! 🎉")
                # prepare chat state (load latest session if exists)
                sessions = load_chat_sessions(uid)
                if sessions:
                    st.session_state.chat_session_id = sessions[0][0]
                    st.session_state.chat_history = sessions[0][1]
                else:
                    st.session_state.chat_session_id = create_chat_session(uid)
                    st.session_state.chat_history = []
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        reg_username = st.text_input("Username (Register)", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register", key="reg_btn"):
            if create_user(reg_username, reg_password):
                st.success("✅ Account created! Please login from the Login tab.")
            else:
                st.error("⚠️ Username already exists. Try another.")
    st.stop()

# =========================
# 3) Sidebar + Navigation
# =========================
with st.sidebar:
    st.success(f"👤 Logged in as {st.session_state['username']}")
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state.clear()
        st.rerun()

    selected = option_menu(
        'Disease Prediction System',
        [
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Parkinson’s Prediction',
            'HealthBot Assistant',
            'Upload Health Report',
            'Past Predictions',       # NEW PAGE
        ],
        icons=['activity','heart','brain','robot','file-earmark-arrow-up','clock-history'],
        default_index=0
    )

# =========================
# 4) OCR Utility
# =========================
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

# =========================
# 5) Redirect Handling
# =========================
if "redirect_to" in st.session_state and st.session_state["redirect_to"]:
    selected = st.session_state["redirect_to"]
    st.session_state["redirect_to"] = None

# =========================
# 6) Diabetes Prediction
# =========================
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
        try:
            user_input_d = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), int(Age)
            ]
        except ValueError:
            st.error("⚠️ Please fill all fields with valid numeric values.")
        else:
            pred = diabetes_model.predict([user_input_d])
            if pred[0] == 1:
                st.error('The person is likely to have diabetes.')
                status = 'likely to have diabetes'
            else:
                st.success('The person is not diabetic.')
                status = 'not diabetic'
            st.session_state['last_prediction'] = {
                'disease': 'Diabetes',
                'input': user_input_d,
                'result': status
            }
            save_prediction(st.session_state.user_id, "Diabetes", user_input_d, status)

# =========================
# 7) Heart Disease Prediction
# =========================
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
        try:
            user_input_h = [
                int(age), int(sex), int(cp), int(trestbps), int(chol),
                int(fbs), int(restecg), int(thalach), int(exang),
                float(oldpeak), int(slope), int(ca), int(thal)
            ]
        except ValueError:
            st.error("⚠️ Please fill all fields with valid numeric values.")
        else:
            pred = heart_model.predict([user_input_h])
            if pred[0] == 1:
                st.error('The person is likely to have heart disease.')
                status = 'likely to have heart disease'
            else:
                st.success('The person does not have any heart disease.')
                status = 'does not have any heart disease'
            st.session_state['last_prediction'] = {
                'disease': 'Heart Disease',
                'input': user_input_h,
                'result': status
            }
            save_prediction(st.session_state.user_id, "Heart Disease", user_input_h, status)

# =========================
# 8) Parkinson’s Prediction
# =========================
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
        except ValueError:
            st.error("⚠️ Please fill all fields with valid numeric values.")
        else:
            pred = parkinsons_model.predict([user_input])
            if pred[0] == 1:
                st.error("The person likely has Parkinson’s Disease.")
                status = "likely to have Parkinson’s Disease"
            else:
                st.success("The person is healthy.")
                status = "does not have Parkinson’s Disease"

            st.session_state['last_prediction'] = {
                'disease': "Parkinson’s Disease",
                'input': user_input,
                'result': status
            }
            save_prediction(st.session_state.user_id, "Parkinson’s Disease", user_input, status)

# =========================
# 9) HealthBot Assistant
# =========================
if selected == 'HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    # Keep your Gemini setup (unchanged)
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("⚠️ Gemini API key missing or invalid. Please check your configuration.")
        st.stop()

    # Ensure there is a chat session id & history in state
    if "chat_session_id" not in st.session_state:
        sessions = load_chat_sessions(st.session_state.user_id)
        if sessions:
            st.session_state.chat_session_id = sessions[0][0]
            st.session_state.chat_history = sessions[0][1]
        else:
            st.session_state.chat_session_id = create_chat_session(st.session_state.user_id)
            st.session_state.chat_history = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Auto-reply if OCR uploaded (kept exactly as before, but we also persist to DB) ---
    last_pred = st.session_state.get("last_prediction", None)
    if isinstance(last_pred, dict) and last_pred.get("disease") == "General Report":
        report_text = last_pred["result"]
        if not any(msg.get("content") == report_text for msg in st.session_state.chat_history):
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
            # persist current chat
            save_chat_messages(st.session_state.chat_session_id, st.session_state.chat_history)
            st.rerun()

    # --- Sidebar controls for chats (list, load, new, clear, delete) ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Chat Sessions")
        sessions = load_chat_sessions(st.session_state.user_id)
        # List sessions with load buttons
        if sessions:
            for idx, (cid, msgs, created, updated) in enumerate(sessions, start=1):
                label = f"Session #{idx} • Updated: {updated}"
                cols = st.columns([4,1,1])
                with cols[0]:
                    st.caption(label)
                with cols[1]:
                    if st.button("Load", key=f"load_chat_{cid}"):
                        st.session_state.chat_session_id = cid
                        st.session_state.chat_history = msgs
                        st.rerun()
                with cols[2]:
                    if st.button("🗑️", key=f"del_chat_{cid}"):
                        delete_chat(cid)
                        # if deleting current, reset to latest or create new
                        if st.session_state.get("chat_session_id") == cid:
                            remaining = load_chat_sessions(st.session_state.user_id)
                            if remaining:
                                st.session_state.chat_session_id = remaining[0][0]
                                st.session_state.chat_history = remaining[0][1]
                            else:
                                st.session_state.chat_session_id = create_chat_session(st.session_state.user_id)
                                st.session_state.chat_history = []
                        st.rerun()
        else:
            st.info("No chats yet.")

        if st.button("➕ New Chat", key="new_chat_btn"):
            st.session_state.chat_session_id = create_chat_session(st.session_state.user_id)
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🧹 Clear Current Chat", key="clear_chat_btn"):
            st.session_state.chat_history = []
            save_chat_messages(st.session_state.chat_session_id, [])
            st.rerun()

    # --- Show current chat history (UI kept as you wrote) ---
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

    # --- Input field + reply (same logic, plus persistence) ---
    user_message = st.chat_input("💬 Type your message...")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Add last prediction context (kept)
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
        # persist to DB
        save_chat_messages(st.session_state.chat_session_id, st.session_state.chat_history)
        st.rerun()

# =========================
# 10) Upload Health Report
# =========================
if selected == "Upload Health Report":
    st.title("📑 Upload Health Report for OCR Analysis")
    uploaded_file = st.file_uploader("Upload health report image", type=["png", "jpg", "jpeg"], key="ocr_upload")
    if uploaded_file is not None:
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_file)
        st.subheader("📄 Extracted Text")
        st.text(extracted_text)
        # trigger HealthBot auto-analysis (kept)
        st.session_state['last_prediction'] = {
            'disease': "General Report",
            'input': [],
            'result': extracted_text
        }
        st.session_state["redirect_to"] = "HealthBot Assistant"
        st.rerun()

# =========================
# 11) Past Predictions (NEW)
# =========================
if selected == "Past Predictions":
    st.title("📜 Past Predictions History")
    preds = load_predictions(st.session_state.user_id)

    # Optional: quick filter
    diseases = ["All", "Diabetes", "Heart Disease", "Parkinson’s Disease"]
    filt = st.selectbox("Filter by disease", diseases, index=0, key="pred_filter")

    shown = [
        (d, vals, res, ts) for (d, vals, res, ts) in preds
        if filt == "All" or d == filt
    ]

    if not shown:
        st.info("No past predictions found.")
    else:
        for i, (d, vals, res, ts) in enumerate(shown, start=1):
            with st.expander(f"{i}. {d} → {res} ({ts})", expanded=False):
                st.write("**Input Values:**")
                st.code(json.dumps(vals, indent=2))
                st.write("**Result:**", res)

