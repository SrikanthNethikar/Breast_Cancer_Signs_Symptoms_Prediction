import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sklearn # Ensure scikit-learn is available for model compatibility

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Early Breast Cancer Risk Prediction")

# --- Constants for File Paths ---
# Define paths relative to your repository's root
MODEL_PATH = "model/breast_cancer_risk_model.pkl"
FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"
# This CSV is used as background data for SHAP explanations
SHAP_BACKGROUND_DATA_PATH = "data/breast_cancer_early_risk.csv"

# --- Helper Functions (with caching for performance) ---

@st.cache_resource
def load_model(path):
    """Loads the pre-trained machine learning model."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please ensure it's in your 'model/' directory on GitHub.")
        st.stop() # Stop the app if the model isn't found
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_feature_columns(path):
    """Loads the list of feature columns the model was trained on."""
    try:
        features = joblib.load(path)
        return features
    except FileNotFoundError:
        st.error(f"Error: Feature columns file not found at {path}. Please ensure it's in your 'model/' directory on GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading feature columns: {e}")
        st.stop()

@st.cache_data
def load_original_data_for_shap(path, feature_cols):
    """
    Loads the original dataset to be used as background data for SHAP.
    It will select only the columns relevant to the model.
    """
    try:
        df = pd.read_csv(path)
        # Ensure the background data only contains the features the model expects
        # and handle any missing columns by filling with 0
        df_for_shap = df.reindex(columns=feature_cols, fill_value=0)
        return df_for_shap
    except FileNotFoundError:
        st.error(f"Error: SHAP background data file not found at {path}. Please ensure it's in your 'data/' directory on GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading SHAP background data: {e}")
        st.stop()

# --- Load Model, Feature Columns, and SHAP Background Data ---
model = load_model(MODEL_PATH)
feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
# Load a sample of the original data for SHAP background.
# It's good practice to use a representative sample if the dataset is very large.
shap_background_data = load_original_data_for_shap(SHAP_BACKGROUND_DATA_PATH, feature_columns)


# --- App Title and Description ---
st.title("Early Breast Cancer Risk Prediction App")
st.markdown("""
    This application uses a machine learning model to estimate early breast cancer risk
    based on various clinical, radiological, and lifestyle factors.
    Please input the patient's data below to get a prediction and an explanation of the model's decision.
""")

st.markdown("---")

# --- Input Section ---
st.header("📝 Patient Data Input")

# Using st.columns for a better layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Mean Cell Measurements")
    mean_radius = st.number_input("Mean Radius (mm)", value=14.0, min_value=0.0, format="%.2f")
    mean_texture = st.number_input("Mean Texture", value=20.0, min_value=0.0, format="%.2f")
    mean_perimeter = st.number_input("Mean Perimeter (mm)", value=90.0, min_value=0.0, format="%.2f")
    mean_area = st.number_input("Mean Area (mm²)", value=600.0, min_value=0.0, format="%.2f")
    mean_smoothness = st.number_input("Mean Smoothness", value=0.1, min_value=0.0, format="%.3f")

with col2:
    st.subheader("📈 Error Metrics")
    mean_compactness = st.number_input("Mean Compactness", value=0.1, min_value=0.0, format="%.3f")
    mean_concavity = st.number_input("Mean Concavity", value=0.1, min_value=0.0, format="%.3f")
    mean_concave_points = st.number_input("Mean Concave Points", value=0.1, min_value=0.0, format="%.3f")
    mean_symmetry = st.number_input("Mean Symmetry", value=0.2, min_value=0.0, format="%.3f")
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06, min_value=0.0, format="%.4f")
    radius_error = st.number_input("Radius Error", value=0.5, min_value=0.0, format="%.3f")
    texture_error = st.number_input("Texture Error", value=1.0, min_value=0.0, format="%.3f")
    perimeter_error = st.number_input("Perimeter Error", value=3.0, min_value=0.0, format="%.3f")
    area_error = st.number_input("Area Error", value=40.0, min_value=0.0, format="%.3f")
    smoothness_error = st.number_input("Smoothness Error", value=0.005, min_value=0.0, format="%.4f")

with col3:
    st.subheader("🧪 Worst Cell Measurements")
    compactness_error = st.number_input("Compactness Error", value=0.02, min_value=0.0, format="%.4f")
    concavity_error = st.number_input("Concavity Error", value=0.03, min_value=0.0, format="%.4f")
    concave_points_error = st.number_input("Concave Points Error", value=0.02, min_value=0.0, format="%.4f")
    symmetry_error = st.number_input("Symmetry Error", value=0.02, min_value=0.0, format="%.4f")
    fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.003, min_value=0.0, format="%.5f")
    worst_radius = st.number_input("Worst Radius (mm)", value=17.0, min_value=0.0, format="%.2f")
    worst_texture = st.number_input("Worst Texture", value=25.0, min_value=0.0, format="%.2f")
    worst_perimeter = st.number_input("Worst Perimeter (mm)", value=110.0, min_value=0.0, format="%.2f")
    worst_area = st.number_input("Worst Area (mm²)", value=800.0, min_value=0.0, format="%.2f")
    worst_smoothness = st.number_input("Worst Smoothness", value=0.15, min_value=0.0, format="%.3f")

st.markdown("---")
st.subheader("🧝 Patient History & Lifestyle")
col4, col5, col6 = st.columns(3)

with col4:
    worst_compactness = st.number_input("Worst Compactness", value=0.3, min_value=0.0, format="%.3f")
    worst_concavity = st.number_input("Worst Concavity", value=0.4, min_value=0.0, format="%.3f")
    worst_concave_points = st.number_input("Worst Concave Points", value=0.2, min_value=0.0, format="%.3f")
    worst_symmetry = st.number_input("Worst Symmetry", value=0.3, min_value=0.0, format="%.3f")
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.09, min_value=0.0, format="%.4f")

with col5:
    likely_malignant = st.selectbox("Initial Diagnosis", options=[0, 1], format_func=lambda x: "Benign (0)" if x == 0 else "Malignant (1)")
    family_history = st.selectbox("Family History of Breast Cancer", options=["No", "Yes"])
    menopause_status = st.selectbox("Menopause Status", options=["Pre", "Post"])
    alcohol = st.selectbox("Alcohol Intake Per Week", options=["Light", "Moderate", "Heavy"])

with col6:
    physical_activity = st.selectbox("Physical Activity Level", options=["Active", "Moderate", "Sedentary"])
    nipple_discharge = st.selectbox("Nipple Discharge", options=["No", "Yes"])
    palpable_lump = st.selectbox("Palpable Lump", options=["No", "Yes"])
    localized_pain = st.selectbox("Localized Breast Pain", options=["No", "Yes"])


# --- Prepare Input for Prediction ---
# Create a dictionary for the input features
input_dict = {
    "mean radius": mean_radius,
    "mean texture": mean_texture,
    "mean perimeter": mean_perimeter,
    "mean area": mean_area,
    "mean smoothness": mean_smoothness,
    "mean compactness": mean_compactness,
    "mean concavity": mean_concavity,
    "mean concave points": mean_concave_points,
    "mean symmetry": mean_symmetry,
    "mean fractal dimension": mean_fractal_dimension,
    "radius error": radius_error,
    "texture error": texture_error,
    "perimeter error": perimeter_error,
    "area error": area_error,
    "smoothness error": smoothness_error,
    "compactness error": compactness_error,
    "concavity error": concavity_error,
    "concave points error": concave_points_error,
    "symmetry error": symmetry_error,
    "fractal dimension error": fractal_dimension_error,
    "worst radius": worst_radius,
    "worst texture": worst_texture,
    "worst perimeter": worst_perimeter,
    "worst area": worst_area,
    "worst smoothness": worst_smoothness,
    "worst compactness": worst_compactness,
    "worst concavity": worst_concavity,
    "worst concave points": worst_concave_points,
    "worst symmetry": worst_symmetry,
    "worst fractal dimension": worst_fractal_dimension,
    "likely_malignant": likely_malignant,
}

# Add one-hot encoded features, setting their value to 1
# All other one-hot encoded columns not selected will implicitly be 0 due to reindex and fill_value=0
input_dict[f"family_history_breast_cancer_{family_history}"] = 1
input_dict[f"menopause_status_{menopause_status}"] = 1
input_dict[f"alcohol_intake_per_week_{alcohol}"] = 1
input_dict[f"physical_activity_level_{physical_activity}"] = 1
input_dict[f"nipple_discharge_{nipple_discharge}"] = 1
input_dict[f"palpable_lump_{palpable_lump}"] = 1
input_dict[f"localized_breast_pain_{localized_pain}"] = 1


# Convert input dictionary to a DataFrame, ensuring column order and presence
# `feature_columns` is loaded from 'model/feature_columns.pkl' and contains all expected column names.
input_encoded = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)


# --- Prediction Button ---
st.markdown("---")
if st.button("🔍 Predict Risk"):
    try:
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"🩺 **High Risk of Early Breast Cancer** (Probability: {prediction_proba[1]:.2f})")
        else:
            st.success(f"✅ **Low Risk of Early Breast Cancer** (Probability: {prediction_proba[0]:.2f})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all input values are valid and compatible with the model.")


# --- SHAP Explanation Button ---
if st.button("🔬 Explain Prediction"):
    try:
        st.subheader("🔍 SHAP Explanation (Top Features Contributing to Prediction)")

        # Initialize SHAP explainer with the trained model and background data
        # Use a sample of the background data if it's very large for performance
        explainer = shap.TreeExplainer(model, data=shap_background_data)

        # Calculate SHAP values for the current input
        # Ensure the input_encoded DataFrame has the same columns as shap_background_data
        shap_values = explainer.shap_values(input_encoded)

        # For binary classification, shap_values is a list of two arrays.
        # We usually explain the positive class (index 1).
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_plot = shap_values[1][0] # Get SHAP values for the positive class (Malignant) for the first instance
            expected_value_to_plot = explainer.expected_value[1] # Get expected value for the positive class
        else:
            # For regression or single-output models, or if only one array is returned
            shap_values_to_plot = shap_values[0] # Get SHAP values for the first instance
            expected_value_to_plot = explainer.expected_value


        # --- SHAP Waterfall Plot ---
        st.markdown("#### How individual features push the prediction from the base value to the output")
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(12, 8)) # Create a matplotlib figure
        shap_explanation = shap.Explanation(
            values=shap_values_to_plot,
            base_values=expected_value_to_plot,
            data=input_encoded.iloc[0], # Pass the actual input data for features
            feature_names=feature_columns # Use the loaded feature names
        )
        shap.plots.waterfall(shap_explanation, show=False, ax=ax_waterfall, max_display=15)
        st.pyplot(fig_waterfall, use_container_width=True)
        plt.clf() # Clear the current figure
        plt.close(fig_waterfall) # Close the figure to free memory


        # --- SHAP Summary Plot (Bar) ---
        st.markdown("#### Overall impact of features on the model's output")
        fig_summary, ax_summary = plt.subplots(figsize=(12, 7)) # Create another matplotlib figure
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-output, use shap_values[1] for positive class
            shap.summary_plot(shap_values[1], shap_background_data, plot_type="bar", show=False, ax=ax_summary, max_display=15)
        else:
            shap.summary_plot(shap_values, shap_background_data, plot_type="bar", show=False, ax=ax_summary, max_display=15)
        st.pyplot(fig_summary, use_container_width=True)
        plt.clf() # Clear the current figure
        plt.close(fig_summary) # Close the figure to free memory

    except Exception as e:
        st.error(f"An error occurred during SHAP explanation: {e}")
        st.info("Please ensure your SHAP background data is compatible with the model and features.")

st.markdown("---")
st.markdown("Developed by Srikanth Nethikar")