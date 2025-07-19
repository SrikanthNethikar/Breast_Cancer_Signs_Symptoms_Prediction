# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer # To get feature names and target names

# --- Load the trained model ---
# Ensure 'decision_tree_model.joblib' is in the same directory as app.py
# This model was created by running model_training.py
try:
    model = joblib.load('decision_tree_model.joblib')
except FileNotFoundError:
    st.error("Error: Model file 'decision_tree_model.joblib' not found.")
    st.error("Please ensure you have run 'model_training.py' to create the model file.")
    st.stop() # Stop the app if model is not found

# --- Get feature names and target names ---
# This ensures consistency with how the model was trained
cancer_data = load_breast_cancer()
feature_names = cancer_data.feature_names
target_names = cancer_data.target_names # 0: malignant, 1: benign

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🔬 Breast Cancer Prediction App")
st.markdown("""
This application predicts whether a breast mass is **Malignant** or **Benign**
based on various diagnostic measurements.

**Please input the values for the patient's features in the sidebar.**
""")

st.markdown("---")

# --- Sidebar for Input Features ---
st.sidebar.header("Input Patient Data (Feature Values)")
st.sidebar.markdown("Adjust the sliders below to enter the measurements for the patient.")

input_data = {}
# Create input fields for each feature in the sidebar
# We'll default the value to the mean of each feature from the original dataset
# for a sensible starting point.
for i, feature in enumerate(feature_names):
    # Determine min, max, and step for sliders based on dataset characteristics
    # This makes the sliders more practical
    min_val = float(cancer_data.data[:, i].min())
    max_val = float(cancer_data.data[:, i].max())
    mean_val = float(cancer_data.data[:, i].mean())

    # Use a number_input for numerical features, with reasonable limits
    input_data[feature] = st.sidebar.slider(
        f"{feature.replace('_', ' ').title()}", # Nicer display name
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=(max_val - min_val) / 1000 # Small step for fine control
    )

# Convert input data dictionary to a Pandas DataFrame
# Ensure the order of columns matches the order used during model training
input_df = pd.DataFrame([input_data], columns=feature_names)

st.subheader("Patient Input Data Preview:")
st.write(input_df)

# --- Prediction Button and Logic ---
st.markdown("---")
st.header("Prediction")

col1, col2 = st.columns([1, 2])

with col1:
    predict_button = st.button("Get Prediction", help="Click to predict the outcome based on inputs")

with col2:
    if predict_button:
        try:
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df) # Probability of each class

            # Get the predicted class name (0 is malignant, 1 is benign)
            predicted_class_name = target_names[prediction[0]].capitalize() # Capitalize for display

            st.success(f"### The model predicts: **{predicted_class_name}**")

            # Display probabilities
            st.write(f"Probability of Malignant (Class 0): `{prediction_proba[0][0]:.4f}`")
            st.write(f"Probability of Benign (Class 1): `{prediction_proba[0][1]:.4f}`")

            if prediction[0] == 0: # Malignant
                st.warning("⚠️ Prediction suggests: **Malignant** - It is strongly recommended to consult a medical professional for further diagnosis.")
            else: # Benign
                st.info("✅ Prediction suggests: **Benign** - It is still recommended to consult a medical professional for confirmation and regular check-ups.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check the input values and ensure the model is loaded correctly.")

st.markdown("---")
st.caption("Disclaimer: This app is for educational and demonstrative purposes only and should not be used for actual medical diagnosis. Always consult with a qualified healthcare professional for any medical concerns.")
