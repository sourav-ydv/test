# -*- coding: utf-8 -*-
"""
Multi Disease Prediction + HealthBot
Updated for OpenAI SDK >= 1.0.0
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI

# ---------------------------------------------------------
# 1️⃣ Load Models
# ---------------------------------------------------------
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# ---------------------------------------------------------
# 2️⃣ App Sidebar Menu
# ---------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        'Multi-Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson’s Prediction', 'HealthBot Assistant'],
        icons=['activity', 'heart', 'brain', 'robot'],
        default_index=0
    )

# ---------------------------------------------------------
# 4️⃣ Diabetes Prediction
# ---------------------------------------------------------
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction using ML")

    Pregnancies = st.number_input("Pregnancies", 0)
    Glucose = st.number_input("Glucose Level", 0)
    BloodPressure = st.number_input("Blood Pressure value", 0)
    SkinThickness = st.number_input("Skin Thickness value", 0)
    Insulin = st.number_input("Insulin Level", 0)
    BMI = st.number_input("BMI value", 0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value", 0.0)
    Age = st.number_input("Age", 0)

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                   Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            st.error('The person is likely to have diabetes.')
        else:
            st.success('The person is not diabetic.')

# ---------------------------------------------------------
# 5️⃣ Heart Disease Prediction
# ---------------------------------------------------------
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 0)
        sex = st.selectbox('Sex (1=Male, 0=Female)', [0, 1])
        cp = st.number_input('Chest Pain types', 0)
        trestbps = st.number_input('Resting Blood Pressure', 0)
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', 0)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0, 1])
        restecg = st.number_input('Resting Electrocardiographic results', 0)
        thalach = st.number_input('Maximum Heart Rate achieved', 0)
    with col3:
        exang = st.selectbox('Exercise Induced Angina (1=True, 0=False)', [0, 1])
        oldpeak = st.number_input('ST depression induced by exercise', 0.0)
        slope = st.number_input('Slope of the peak exercise ST segment', 0)
        ca = st.number_input('Major vessels colored by fluoroscopy', 0)
        thal = st.number_input('thal (0=Normal, 1=Fixed defect, 2=Reversable defect)', 0)

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,
                                                 thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            st.error('The person is likely to have heart disease.')
        else:
            st.success('The person does not have any heart disease.')

# ---------------------------------------------------------
# 6️⃣ Parkinson’s Prediction
# ---------------------------------------------------------
if selected == 'Parkinson’s Prediction':
    st.title("Parkinson’s Disease Prediction using ML")

    # (You can shorten or keep full inputs — this example assumes model expects 22 inputs)
    inputs = []
    for i in range(1, 23):
        inputs.append(st.number_input(f'Feature {i}', 0.0))

    if st.button('Parkinson’s Test Result'):
        park_prediction = parkinsons_model.predict([inputs])
        if park_prediction[0] == 1:
            st.error('The person likely has Parkinson’s Disease.')
        else:
            st.success('The person is healthy.')

# ---------------------------------------------------------
# 7️⃣ HealthBot (New API-compatible)
# ---------------------------------------------------------
if selected == 'HealthBot Assistant':
    st.title("HealthBot — AI Health Assistant")

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.error("Please set your OpenAI API key in Streamlit Secrets.")
        st.stop()

    def ask_openai(prompt_user, system_instructions, max_tokens=300, temp=0.7):
        """Ask OpenAI chat model safely."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt_user},
                ],
                max_tokens=max_tokens,
                temperature=temp,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error contacting OpenAI API: {e}"

    st.write("Ask health questions or discuss your test results safely below:")

    user_input = st.text_area("You:", placeholder="e.g., What does high glucose level mean?")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            system_prompt = (
                "You are a careful, friendly health assistant. "
                "You provide general health explanations and wellness advice. "
                "You never give medical diagnoses. "
                "Encourage users to consult a doctor for any serious concerns."
            )
            reply = ask_openai(user_input, system_prompt)

        st.markdown("### HealthBot:")
        st.write(reply)


