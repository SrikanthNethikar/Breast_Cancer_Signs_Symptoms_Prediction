import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import io
# Removed: import warnings and sklearn.exceptions.InconsistentVersionWarning as they are not needed for scikit-learn==1.2.2

# Set PIL's max image pixels to None to prevent DecompressionBombError
Image.MAX_IMAGE_PIXELS = None 

# --- 1. Load Model and Preprocessors ---
@st.cache_resource # Cache the model loading for performance
def load_model_and_assets():
    """
    Loads the trained machine learning model, expected column names after OHE,
    and SHAP background data.
    """
    try:
        model = pickle.load(open("breast_cancer_early_signs_model.pkl", "rb"))

        with open("model_columns.pkl", "rb") as f:
            expected_input_columns_after_ohe = pickle.load(f)
        
        # --- CRITICAL: Load SHAP background data ---
        # This file must contain a preprocessed sample of your training data
        with open("shap_background_data.pkl", "rb") as f:
            shap_background_data = pickle.load(f)
        
        return model, expected_input_columns_after_ohe, shap_background_data
    
    except FileNotFoundError as e:
        st.error(f"Error loading model or required files: {e}. "
                 f"Please ensure 'breast_cancer_early_signs_model.pkl', "
                 f"'model_columns.pkl', AND 'shap_background_data.pkl' "
                 f"are in the same directory as app.py.")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        # This catches other potential errors during loading,
        # such as the ValueError due to scikit-learn version mismatch
        st.error(f"An unexpected error occurred while loading model assets: {e}. "
                 f"This often indicates a version mismatch between the scikit-learn "
                 f"version used to train/save the model and the version currently installed. "
                 f"Please ensure your scikit-learn version matches the one used for training. "
                 f"You might need to retrain the model or downgrade scikit-learn.")
        st.stop() # Stop the app execution on other loading errors

# --- Unpack the loaded assets, including the new shap_background_data ---
# This function is called once due to @st.cache_resource
model, expected_input_columns_after_ohe, shap_background_data = load_model_and_assets()

# --- Define Features (ensure these match your training data's structure) ---
NUMERICAL_FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

CATEGORICAL_FEATURES_INFO = {
    'family_history_breast_cancer': ['No', 'Yes'],
    'menopause_status': ['Post', 'Pre'],
    'alcohol_intake_per_week': ['Heavy', 'Light', 'Moderate', 'nan'],
    'physical_activity_level': ['Active', 'Moderate', 'Sedentary'],
    'nipple_discharge': ['No', 'Yes'],
    'palpable_lump': ['No', 'Yes'],
    'localized_breast_pain': ['No', 'Yes']
}

# --- Streamlit UI ---
st.title("🩺 Breast Cancer Early Signs Prediction")
st.header("Enter Patient Information")
user_input_raw = {}

st.subheader("Numerical Features") 
for feature in NUMERICAL_FEATURES:
    # Set default values and ranges as appropriate for your data
    if 'radius' in feature or 'perimeter' in feature or 'area' in feature or 'size' in feature:
        user_input_raw[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, max_value=100.0, value=5.0, key=f"num_{feature}")
    elif 'texture' in feature or 'smoothness' in feature or 'compactness' in feature or 'concavity' in feature or 'symmetry' in feature or 'fractal dimension' in feature:
        user_input_raw[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, max_value=1.0, value=0.1, format="%.4f", key=f"num_{feature}")
    else:
        user_input_raw[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, key=f"num_{feature}")

st.subheader("Categorical Features") 
for feature, categories in CATEGORICAL_FEATURES_INFO.items():
    if len(categories) == 2 and 'Yes' in categories and 'No' in categories: 
        checkbox_value = st.checkbox(f"{feature.replace('_', ' ').title()}?", value=False, key=f"cat_{feature}")
        user_input_raw[feature] = 'Yes' if checkbox_value else 'No'
    else:
        user_input_raw[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", options=categories, key=f"cat_{feature}")

if st.button("Predict Risk"):
    # Convert raw user input into a DataFrame
    input_df_raw = pd.DataFrame([user_input_raw])

    # --- Preprocessing to match model's expected format ---
    for col, categories in CATEGORICAL_FEATURES_INFO.items():
        if col in input_df_raw.columns:
            input_df_raw[col] = pd.Categorical(input_df_raw[col], categories=categories)
        else:
            input_df_raw[col] = pd.Categorical([], categories=categories)

    input_df_processed = pd.get_dummies(input_df_raw, columns=list(CATEGORICAL_FEATURES_INFO.keys()))

    # Reindex to ensure ALL expected columns (including OHE ones) are present and in correct order
    final_input_for_prediction = pd.DataFrame(0, index=[0], columns=expected_input_columns_after_ohe)
    for col in input_df_processed.columns:
        if col in final_input_for_prediction.columns:
            final_input_for_prediction[col] = input_df_processed[col].iloc[0]

    # Ensure all columns are float before passing to model/SHAP
    final_input_for_prediction = final_input_for_prediction.astype(float)

    # --- Debugging final input for model (can be removed once confirmed working) ---
    st.write("Debug: Final input to model shape:", final_input_for_prediction.shape)
    st.write("Debug: Final input to model columns (first 10):", final_input_for_prediction.columns.tolist()[:10])
    st.write("Debug: Final input to model data (first row):", final_input_for_prediction.iloc[0])
    st.write("Debug: final_input_for_prediction dtypes:\n", final_input_for_prediction.dtypes)

    # --- Make Prediction ---
    try:
        prediction = model.predict(final_input_for_prediction)[0]
        proba = model.predict_proba(final_input_for_prediction)[0][1] # Probability of positive class (1)

        st.success(f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        st.info(f"Risk Probability Score: {proba:.2f}")

        # --- SHAP Explanation ---
        st.subheader("Feature Contribution (SHAP Waterfall Plot)")
        
        # Initialize explainer with the loaded model and background data
        explainer = shap.TreeExplainer(model, shap_background_data) 

        # Calculate SHAP values for the current input
        shap_values = explainer(final_input_for_prediction, check_additivity=False)

        # --- CRITICAL FIX: Robust extraction of SHAP values and expected value ---
        st.write("Debug: shap_values.values shape:", shap_values.values.shape)
        st.write("Debug: shap_values.base_values shape:", shap_values.base_values.shape)
        st.write("Debug: explainer.expected_value:", explainer.expected_value)

        # Extract SHAP values for the single instance
        # If shap_values.values is 3D (instance, feature, class), select class 1
        if shap_values.values.ndim == 3 and shap_values.values.shape[2] > 1:
            shap_val = shap_values.values[0, :, 1] # Select first instance, all features, class 1
        else:
            # Otherwise, assume it's 2D (instance, feature) or 1D (feature)
            # and the values are directly for the output being explained
            shap_val = shap_values.values[0] if shap_values.values.ndim > 1 else shap_values.values

        # Extract expected value
        # If explainer.expected_value is an array and has more than one element, select for class 1
        if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim > 0 and explainer.expected_value.size > 1:
            expected_val = explainer.expected_value[1] # Expected value for class 1
        else:
            # Otherwise, assume it's a scalar or a single-element array, take the first/only value
            expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim > 0 else explainer.expected_value


        plot_feature_names = final_input_for_prediction.columns.tolist()

        # --- Debugging SHAP values (can be removed once confirmed working) ---
        st.write("Debug: SHAP values (first 5):", shap_val[:5])
        st.write("Debug: Expected Value:", expected_val)
        if not isinstance(shap_val, np.ndarray) or shap_val.size == 0 or np.all(np.isclose(shap_val, 0)):
            st.warning("Debug: SHAP values are all zeros or empty. Plot may appear blank. (This might indicate an issue with SHAP calculation or model output.)")

        # Clear previous plot to prevent overlap and manage memory
        plt.clf()

        # Temporarily set Matplotlib's default figure size for the plot
        original_figsize = plt.rcParams['figure.figsize']
        plt.rcParams['figure.figsize'] = (10, 8) 

        # Generate the SHAP waterfall plot
        fig = shap.plots._waterfall.waterfall_legacy(
            expected_val,
            shap_val,
            feature_names=plot_feature_names,
            max_display=10, 
            show=False 
        )

        plt.tight_layout() 

        # Save figure to BytesIO with low DPI and display via st.image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight') 
        buf.seek(0) 

        # --- Debugging Image Buffer (can be removed once confirmed working) ---
        st.write("Debug: Size of image buffer (bytes):", len(buf.getvalue()))
        if len(buf.getvalue()) == 0:
            st.error("Debug: Image buffer is empty. Plot failed to save correctly.")

        st.image(buf.read(), use_container_width=True) 
        buf.close() 

        # IMPORTANT: Close the Matplotlib figure to free its memory
        plt.close(fig)
        plt.rcParams['figure.figsize'] = original_figsize 

    except Exception as e:
        st.error(f"An error occurred during prediction or SHAP explanation: {e}")
        st.warning("Please check your input values and ensure your model and SHAP background data are compatible.")

st.markdown("---")
st.caption("Developed with Streamlit and SHAP for educational purposes.")