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
    st.markdown("---")
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

            # attach column names for each model
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

        # 🧹 Reset input box after sending
        st.session_state.chat_input = ""

        # Rerun to refresh UI
        st.rerun()
