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
    # Users table
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT)""")
    # History table
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
        login_username = st.text_input("Username (Login)")
        login_password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_id = login_user(login_username, login_password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["username"] = login_username
                st.success(f"Welcome back, {login_username}! 🎉")
                # Load history if exists
                past = load_history(user_id)
                if past:
                    st.session_state["chat_history"] = ast.literal_eval(past[0][3])
                else:
                    st.session_state["chat_history"] = []
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        reg_username = st.text_input("Username (Register)")
        reg_password = st.text_input("Password", type="password")
        if st.button("Register"):
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
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction',
         'Parkinson’s Prediction', 'HealthBot Assistant', 'Upload Health Report'],
        icons=['activity', 'heart', 'brain', 'robot', 'file-earmark-arrow-up'],
        default_index=0
    )

    # Show history
    st.subheader("📜 Recent History")
    history = load_history(st.session_state["user_id"])
    if history:
        for h in history:
            st.markdown(f"- **{h[0]}** → {h[2]} ({h[4]})")

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
        save_history(st.session_state["user_id"], "Diabetes", user_input_d, diab_status, st.session_state.get("chat_history", []))

# ---------------------------------------------------------
# 6️⃣ Heart Disease Prediction
# ---------------------------------------------------------
# (same logic as before, add save_history after prediction)

# ---------------------------------------------------------
# 7️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
# (same logic as before, add save_history after prediction)

# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='background:#1e1e1e;padding:10px;border-radius:12px;margin:5px;text-align:right;color:#fff;'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#2b313e;padding:10px;border-radius:12px;margin:5px;text-align:left;color:#e2e2e2;'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # Chat input
    user_message = st.chat_input("💬 Type your message...")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        reply = f"(Gemini/AI reply to: {user_message})"  # Placeholder for API
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_history(st.session_state["user_id"], "Chat", [], "Chat Update", st.session_state["chat_history"])
        st.rerun()

# ---------------------------------------------------------
# 9️⃣ Upload Health Report (OCR → Chatbot only)
# ---------------------------------------------------------
if selected == "Upload Health Report":
    st.title("📑 Upload Health Report for OCR Analysis")

    uploaded_file = st.file_uploader("Upload health report image", type=["png", "jpg", "jpeg"])
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
