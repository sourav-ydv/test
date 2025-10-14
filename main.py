# ---------------------------------------------------------
# 8️⃣ HealthBot Assistant (Gemini-Only Chatbot with Enter-to-Send)
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("🤖 AI HealthBot Assistant")

    # --- Gemini Setup ---
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("⚠️ Gemini API key missing or invalid. Please check your configuration.")
        st.stop()

    # --- Initialize Chat Memory ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # --- Chat Display Container ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#1e1e1e;padding:10px 15px;border-radius:12px;"
                    f"margin:8px 0;text-align:right;color:#fff;'>"
                    f"🧑 <b>You:</b> {msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#2b313e;padding:10px 15px;border-radius:12px;"
                    f"margin:8px 0;text-align:left;color:#e2e2e2;'>"
                    f"🤖 <b>HealthBot:</b> {msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # --- Function: Send Message ---
    def handle_send():
        user_text = st.session_state.chat_input.strip()
        if not user_text:
            return

        st.session_state.chat_history.append({"role": "user", "content": user_text})

        # --- System Prompt ---
        system_prompt = (
            "You are a helpful and knowledgeable AI health assistant named HealthBot. "
            "Provide general information on health, wellness, exercise, and diet. "
            "Avoid giving any medical prescriptions or diagnoses. "
            "Encourage users to consult professionals for medical concerns."
        )

        # --- Add Disease Prediction Context ---
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

            input_with_names = "\n".join([f"{c}: {v}" for c, v in zip(columns, values)])

            user_context = (
                f"\nUser recently tested for {disease}.\n"
                f"Input details:\n{input_with_names}\n"
                f"Prediction result: {last_pred['result']}\n"
                "Give lifestyle advice, diet tips, or precautions related to this data."
            )

        full_prompt = f"{system_prompt}\n{user_context}\n\nUser Question: {user_text}"

        # --- Gemini Response ---
        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(full_prompt)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Gemini API error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # ✅ Clear input safely
        st.session_state.chat_input = ""

    # --- Fixed Input Box at Bottom with Enter-to-Send ---
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 80px !important;
        }
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #111;
            padding: 12px;
            border-top: 1px solid #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        user_input = st.text_area(
            "💬 Type your message...",
            key="chat_input",
            height=80,
            placeholder="Ask about diet, fitness, or your health data...",
            label_visibility="collapsed"
        )

        # Detect Enter key press
        if user_input and st.session_state.chat_input == user_input:
            handle_send()

    # Optional Clear Chat button
    st.button("🧹 Clear Chat", use_container_width=True, on_click=lambda: st.session_state.update({"chat_history": [], "chat_input": ""}))
