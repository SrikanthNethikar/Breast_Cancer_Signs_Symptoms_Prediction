import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load model
with open("Breast_Cancer_Signs_Symptoms_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧬 Breast Cancer Early Signs Prediction")

# Sample input form
with st.form("input_form"):
    st.subheader("Enter Patient Data")
    radius = st.number_input("Mean Radius", min_value=0.0)
    texture = st.number_input("Mean Texture", min_value=0.0)
    symmetry = st.number_input("Symmetry", min_value=0.0)
    # 🔁 Add all other inputs used in your model below as needed
    submitted = st.form_submit_button("Predict")

if submitted:
    # Collect input into DataFrame
    input_data = pd.DataFrame(
        [[radius, texture, symmetry]],
        columns=["mean_radius", "mean_texture", "symmetry"]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")

    # SHAP Explanation
    st.subheader("🔍 Feature Contribution (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot()

    # Optionally show pre-generated SHAP force plot
    with open("shap_outputs/shap_force_plot_patient_10.html", "r") as f:
        html_content = f.read()
    components.html(html_content, height=300)