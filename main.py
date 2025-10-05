# -*- coding: utf-8 -*-
"""
Stable Multi-Disease Prediction + Chatbot
(uses Hugging Face Flan-T5, cached; chat persists after prediction,
summary button, refuses off-topic queries, and strips prompt echoes)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from transformers import pipeline

# -------------------------
# Config: change model here if you want
# -------------------------
MODEL_NAME = "google/flan-t5-small"   # change to flan-t5-base for stronger answers (heavier)

# -------------------------
# Load & cache the HF model once (very important)
# -------------------------
@st.cache_resource
def load_chat_model():
    return pipeline(
        "text2text-generation",
        model=MODEL_NAME
    )

chatbot = load_chat_model()

# -------------------------
# Helper: clean model output
# -------------------------
def _clean_response(generated_text: str, user_query: str, prompt_prefix: str) -> str:
    """
    Remove accidental echoing of prompt or labels and repetitive prefixes.
    """
    res = generated_text.strip()

    # If model repeated the prompt literally, drop that prefix
    if prompt_prefix and prompt_prefix.strip() and prompt_prefix in res:
        res = res.replace(prompt_prefix, "").strip()

    # Remove leading "Answer:", "A:", "Response:" and occasionally the user query itself
    prefixes = ["Answer:", "A:", "Response:", "Response -", "Reply:", user_query]
    # strip any of these if they appear at the start
    lowered = res.lower()
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if p and lowered.startswith(p.lower()):
                res = res[len(p):].strip()
                lowered = res.lower()
                changed = True

    # Remove repeated phrases (common small-loop artifacts). Keep it conservative.
    # For extreme repetition like "It is the condition ... It is the condition ...", truncate to first 2 repeats.
    # Simple heuristic: if res contains the same sentence repeated 3+ times, collapse it.
    parts = [s.strip() for s in res.split('. ') if s.strip()]
    if len(parts) > 4:
        # detect repetition of first sentence
        first = parts[0]
        repeats = sum(1 for p in parts if p == first)
        if repeats >= 3:
            # keep first occurrence + next few distinct sentences
            unique_rest = []
            for p in parts:
                if p != first or len(unique_rest) > 0:
                    unique_rest.append(p)
            res = ". ".join([first] + unique_rest[:4])
    return res.strip()

# -------------------------
# Helper: generate answer (deterministic, no sampling, beam search)
# -------------------------
def get_chatbot_response(disease: str, diagnosis: str, user_query: str = "", summary: bool = False) -> str:
    """
    disease: "Diabetes", "Heart Disease", "Parkinson's Disease"
    diagnosis: e.g., "The person has Diabetes"  (a short string)
    user_query: user's question (empty when summary=True)
    summary: if True, produce a concise summary (few bullets / short paragraphs)
    """
    # small vocabulary of terms that indicate a health question (used to permit/deny off-topic)
    allowed_terms = [disease.lower(), "symptom", "symptoms", "treat", "treatment", "diet", "exercise",
                     "insulin", "medication", "medicine", "blood", "sugar", "cholesterol", "risk",
                     "precaution", "lifestyle", "manage", "control", "reduce", "severity", "care"]

    # If user_query is provided, apply a simple keyword check to refuse unrelated queries.
    if not summary and user_query:
        q_lower = user_query.lower()
        if not any(tok in q_lower for tok in allowed_terms):
            return f"Sorry—I can answer questions only about {disease}. Please ask about symptoms, treatment, lifestyle, or precautions related to {disease}."

    # Build concise instruction-style prompt and use explicit "Answer:" cue so model knows where to start generating
    if summary:
        prompt = (
            f"Summarize briefly (3-6 short bullets) the diagnosis and practical advice for a patient with {disease}."
            f" Diagnosis: {diagnosis}."
            " Include 1–2 practical lifestyle steps, 1 precaution, and end with 'Consult a doctor.'"
            "\nAnswer:"
        )
    else:
        # keep prompt small and focused; the "Q:" / "A:" pattern helps Flan-T5 produce just the answer
        prompt = (
            f"Context: Patient diagnosed with {disease}. Diagnosis: {diagnosis}.\n"
            f"Q: {user_query}\nA:"
        )

    # Generate using deterministic beams, avoid sampling to reduce odd outputs / repetition
    # Use no_repeat_ngram_size to discourage verbatim repetition
    try:
        out = chatbot(
            prompt,
            # generation params
            max_length=180,          # total tokens in output
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        raw = out[0]["generated_text"]
    except Exception as e:
        # fallback simple call if pipeline signature differs in some envs
        out = chatbot(prompt, max_length=180)
        raw = out[0]["generated_text"]

    cleaned = _clean_response(raw, user_query, prompt_prefix=prompt)
    return cleaned

# -------------------------
# Load ML models (pickles)
# -------------------------
# NOTE: keep these filenames matching your saved .sav files
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.sav", "rb"))

# -------------------------
# Streamlit UI + state setup
# -------------------------
st.set_page_config(page_title="Disease Predictor + Chatbot", layout="wide")
st.title("Multiple Disease Prediction System + Medical Chatbot")

with st.sidebar:
    page = option_menu("Menu", ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
                       icons=["activity", "heart", "person"], menu_icon="cast", default_index=0)

# Initialize shared session state for chat (one chat history for the currently diagnosed disease)
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = ""           # short result text
if "disease_name" not in st.session_state:
    st.session_state.disease_name = ""        # "Diabetes", etc.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # list of {"role": "user"/"assistant", "content": "..."}
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = {}         # store last numeric inputs if needed

# ---------- Diabetes Page ----------
if page == "Diabetes Prediction":
    st.header("🩸 Diabetes Prediction")
    # free-form inputs (user types numbers)
    Pregnancies = st.text_input("Number of Pregnancies", value=st.session_state.last_inputs.get("Pregnancies", ""))
    Glucose = st.text_input("Glucose Level (mg/dL)", value=st.session_state.last_inputs.get("Glucose", ""))
    BloodPressure = st.text_input("Blood Pressure (mm Hg)", value=st.session_state.last_inputs.get("BloodPressure", ""))
    SkinThickness = st.text_input("Skin Thickness (mm)", value=st.session_state.last_inputs.get("SkinThickness", ""))
    Insulin = st.text_input("Insulin Level (mu U/ml)", value=st.session_state.last_inputs.get("Insulin", ""))
    BMI = st.text_input("BMI (Body Mass Index)", value=st.session_state.last_inputs.get("BMI", ""))
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", value=st.session_state.last_inputs.get("DiabetesPedigreeFunction", ""))
    Age = st.text_input("Age (years)", value=st.session_state.last_inputs.get("Age", ""))

    if st.button("🔍 Get Diabetes Test Result"):
        try:
            inputs = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            pred = diabetes_model.predict([inputs])
            if pred[0] == 1:
                st.session_state.diagnosis = "Has Diabetes"
            else:
                st.session_state.diagnosis = "No Diabetes"
            st.session_state.disease_name = "Diabetes"
            st.session_state.last_inputs.update({
                "Pregnancies": Pregnancies, "Glucose": Glucose, "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness, "Insulin": Insulin, "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age
            })
            # optionally clear chat history when new diagnosis differs
            st.success(f"Diagnosis: {st.session_state.diagnosis}")

        except Exception:
            st.session_state.diagnosis = "⚠️ Invalid input — please enter numeric values only."
            st.session_state.disease_name = "Diabetes"
            st.error(st.session_state.diagnosis)

# ---------- Heart Disease Page ----------
if page == "Heart Disease Prediction":
    st.header("❤️ Heart Disease Prediction")
    age = st.text_input("Age", value=st.session_state.last_inputs.get("age", ""))
    sex = st.text_input("Sex (0=Female,1=Male)", value=st.session_state.last_inputs.get("sex", ""))
    cp = st.text_input("Chest pain type", value=st.session_state.last_inputs.get("cp", ""))
    trestbps = st.text_input("Resting BP", value=st.session_state.last_inputs.get("trestbps", ""))
    chol = st.text_input("Cholesterol", value=st.session_state.last_inputs.get("chol", ""))
    fbs = st.text_input("Fasting Blood Sugar >120 mg/dl (0/1)", value=st.session_state.last_inputs.get("fbs", ""))
    restecg = st.text_input("Rest ECG", value=st.session_state.last_inputs.get("restecg", ""))
    thalach = st.text_input("Max Heart Rate", value=st.session_state.last_inputs.get("thalach", ""))
    exang = st.text_input("Exercise Induced Angina (0/1)", value=st.session_state.last_inputs.get("exang", ""))
    oldpeak = st.text_input("ST depression (oldpeak)", value=st.session_state.last_inputs.get("oldpeak", ""))
    slope = st.text_input("Slope", value=st.session_state.last_inputs.get("slope", ""))
    ca = st.text_input("Major vessels (0-3)", value=st.session_state.last_inputs.get("ca", ""))
    thal = st.text_input("Thal (0-3)", value=st.session_state.last_inputs.get("thal", ""))

    if st.button("🔍 Get Heart Disease Test Result"):
        try:
            inputs = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]
            pred = heart_model.predict([inputs])
            st.session_state.diagnosis = "Has Heart Disease" if pred[0] == 1 else "No Heart Disease"
            st.session_state.disease_name = "Heart Disease"
            st.session_state.last_inputs.update({
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            })
            st.success(f"Diagnosis: {st.session_state.diagnosis}")
        except Exception:
            st.session_state.diagnosis = "⚠️ Invalid input — please enter numeric values only."
            st.session_state.disease_name = "Heart Disease"
            st.error(st.session_state.diagnosis)

# ---------- Parkinson's Page ----------
if page == "Parkinson's Prediction":
    st.header("🧠 Parkinson's Disease Prediction (simplified inputs)")
    Fo = st.text_input("MDVP:Fo", value=st.session_state.last_inputs.get("Fo", ""))
    Fhi = st.text_input("MDVP:Fhi", value=st.session_state.last_inputs.get("Fhi", ""))
    Flo = st.text_input("MDVP:Flo", value=st.session_state.last_inputs.get("Flo", ""))
    Jitter_percent = st.text_input("MDVP:Jitter (%)", value=st.session_state.last_inputs.get("Jitter_percent", ""))
    Shimmer = st.text_input("MDVP:Shimmer", value=st.session_state.last_inputs.get("Shimmer", ""))
    HNR = st.text_input("HNR", value=st.session_state.last_inputs.get("HNR", ""))

    if st.button("🔍 Get Parkinson's Test Result"):
        try:
            inputs = [float(Fo), float(Fhi), float(Flo), float(Jitter_percent), float(Shimmer), float(HNR)]
            pred = parkinsons_model.predict([inputs])
            st.session_state.diagnosis = "Has Parkinson's Disease" if pred[0] == 1 else "No Parkinson's Disease"
            st.session_state.disease_name = "Parkinson's Disease"
            st.session_state.last_inputs.update({
                "Fo": Fo, "Fhi": Fhi, "Flo": Flo, "Jitter_percent": Jitter_percent,
                "Shimmer": Shimmer, "HNR": HNR
            })
            st.success(f"Diagnosis: {st.session_state.diagnosis}")
        except Exception:
            st.session_state.diagnosis = "⚠️ Invalid input — please enter numeric values only."
            st.session_state.disease_name = "Parkinson's Disease"
            st.error(st.session_state.diagnosis)

# -------------------------
# GLOBAL Chatbot area (shows after any diagnosis stored)
# -------------------------
st.divider()
st.header("Assistant")

if st.session_state.diagnosis:
    st.subheader(f"{st.session_state.disease_name} Assistant Chatbot")
    # Ensure chat history exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Summary button
    if st.button("📋 Summary of Condition", key="summary_global"):
        # generate summary and append to chat history
        summary_text = get_chatbot_response(st.session_state.disease_name, st.session_state.diagnosis, summary=True)
        st.session_state.chat_history.append({"role": "assistant", "content": summary_text})
        st.chat_message("assistant").write(summary_text)

    # Chat input for disease-specific questions
    if user_q := st.chat_input("Ask a question about this condition..."):
        # append user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        st.chat_message("user").write(user_q)
        # generate reply
        reply = get_chatbot_response(st.session_state.disease_name, st.session_state.diagnosis, user_query=user_q, summary=False)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

else:
    st.info("Run a prediction above to enable the disease assistant chatbot.")
