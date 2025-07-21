import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sklearn # Make sure this is here to check version if needed, though we already got it

# --- Configuration ---
# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Breast Cancer Prediction App")

# --- Constants ---
# Define the paths to your model and data (adjust if different)
MODEL_PATH = 'model/trained_model.pkl'  # Ensure this path is correct
DATA_PATH = 'data/breast_cancer_data.csv'  # Ensure this path is correct

# Define the features your model expects (IMPORTANT: MUST MATCH YOUR TRAINING DATA)
FEATURES = [
    'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
]
TARGET_COLUMN = 'Class' # The name of your target column in the original dataset

# --- Helper Functions (using Streamlit's caching for performance) ---

@st.cache_resource # Use st.cache_resource for models and large objects
def load_model(path):
    """Loads the pre-trained model."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please check the path and ensure it's in your repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data # Use st.cache_data for dataframes
def load_data(path):
    """Loads the dataset for display and SHAP background."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {path}. Please check the path and ensure it's in your repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# --- Load Model and Data ---
model = load_model(MODEL_PATH)
df_original = load_data(DATA_PATH)

# Prepare data for SHAP: exclude target and any non-feature columns
# IMPORTANT: Ensure df_for_shap contains only the features your model was trained on
df_for_shap = df_original[FEATURES]


# --- App Title and Description ---
st.title("Breast Cancer Early Signs & Symptoms Prediction")
st.markdown("""
    This application predicts the likelihood of breast cancer based on various cell characteristics.
    Please enter the values for the features below to get a prediction and an explanation of the model's output.
""")

# --- Input Section ---
st.header("Input Patient Data")

# Using st.columns for a better layout for inputs
col1, col2, col3 = st.columns(3)

input_data = {}
with col1:
    input_data['Clump Thickness'] = st.slider('Clump Thickness (1-10)', 1, 10, 5)
    input_data['Uniformity of Cell Size'] = st.slider('Uniformity of Cell Size (1-10)', 1, 10, 5)
    input_data['Uniformity of Cell Shape'] = st.slider('Uniformity of Cell Shape (1-10)', 1, 10, 5)
with col2:
    input_data['Marginal Adhesion'] = st.slider('Marginal Adhesion (1-10)', 1, 10, 5)
    input_data['Single Epithelial Cell Size'] = st.slider('Single Epithelial Cell Size (1-10)', 1, 10, 5)
    input_data['Bare Nuclei'] = st.slider('Bare Nuclei (1-10)', 1, 10, 5)
with col3:
    input_data['Bland Chromatin'] = st.slider('Bland Chromatin (1-10)', 1, 10, 5)
    input_data['Normal Nucleoli'] = st.slider('Normal Nucleoli (1-10)', 1, 10, 5)
    input_data['Mitoses'] = st.slider('Mitoses (1-10)', 1, 10, 5)

# Convert input_data to a DataFrame for prediction
input_df = pd.DataFrame([input_data])


# --- Prediction and Explanation Section ---
st.header("Prediction Results and Explanation")

if st.button("Predict"):
    try:
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_df)[0]
        # Get the predicted class (0 for Benign, 1 for Malignant assuming your model outputs this)
        predicted_class = np.argmax(prediction_proba)

        st.subheader("Prediction:")
        if predicted_class == 0:
            st.success(f"**Prediction: Benign** (Probability: {prediction_proba[0]:.2f})")
        else:
            st.error(f"**Prediction: Malignant** (Probability: {prediction_proba[1]:.2f})")

        st.markdown("---") # Separator

        st.subheader("Model Explanation (SHAP Values)")

        # Initialize SHAP explainer
        # For tree-based models, TreeExplainer is efficient
        explainer = shap.TreeExplainer(model, data=df_for_shap) # Pass background data for better explanations

        # Calculate SHAP values for the current input
        # Ensure the input_df has feature names matching df_for_shap
        shap_values = explainer.shap_values(input_df)

        # --- Debug SHAP output for clarity ---
        st.write(f"Debug: shap_values type: {type(shap_values)}")
        if isinstance(shap_values, list): # For multi-output models (e.g., multi-class or binary with two columns)
            st.write(f"Debug: shap_values (list of arrays) length: {len(shap_values)}")
            st.write(f"Debug: shap_values[0] shape: {shap_values[0].shape}")
            st.write(f"Debug: shap_values[1] shape: {shap_values[1].shape}")
            # Assuming binary classification where shap_values[1] is for the positive class
            shap_values_to_plot = shap_values[1]
            expected_value_to_plot = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            st.write(f"Debug: Explainer expected_value (for class 1): {expected_value_to_plot}")
        else: # For single-output models (e.g., regression or binary as one column)
            st.write(f"Debug: shap_values shape: {shap_values.shape}")
            shap_values_to_plot = shap_values
            expected_value_to_plot = explainer.expected_value
            st.write(f"Debug: Explainer expected_value: {expected_value_to_plot}")

        # Create a SHAP Explanation object for the waterfall plot
        # Use the correct SHAP values and base value (expected_value)
        # Ensure feature_names are passed correctly
        e = shap.Explanation(
            values=shap_values_to_plot[0], # Take the first (and only) instance
            base_values=expected_value_to_plot,
            data=input_df.iloc[0], # Pass the actual input data for features
            feature_names=FEATURES
        )

        st.subheader("Feature Contribution (SHAP Waterfall Plot)")

        # --- THE FIX FOR `use_container_width` ERROR ---
        # Create a matplotlib figure to draw the SHAP plot on
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figsize as needed for better readability

        # Generate the SHAP waterfall plot on the created axes
        # Pass 'show=False' to prevent shap from trying to display it itself
        # and 'ax=ax' to tell it to plot on our specific matplotlib axes.
        shap.plots.waterfall(e, show=False, ax=ax)

        # Display the matplotlib figure in Streamlit with use_container_width
        st.pyplot(fig, use_container_width=True)

        # Clear the current figure to prevent plots from overlapping in subsequent runs
        plt.clf()
        plt.close(fig)
        # --- END OF FIX ---


        st.subheader("Summary of Feature Importance (SHAP Bar Plot)")
        # Calculate SHAP values for a sample of the data for the summary plot
        # It's better to use a representative sample, not just one instance
        # If your df_for_shap is large, sample it for performance on Streamlit Cloud
        if len(df_for_shap) > 1000: # Example: take a sample if data is too big
            sample_df = df_for_shap.sample(1000, random_state=42)
        else:
            sample_df = df_for_shap

        explainer_summary = shap.TreeExplainer(model, data=sample_df)
        shap_values_summary = explainer_summary.shap_values(sample_df)

        fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
        # Assuming binary classification, show importance for the positive class (index 1)
        # If not multi-output or regression, just use shap_values_summary directly
        if isinstance(shap_values_summary, list):
            shap.summary_plot(shap_values_summary[1], sample_df, plot_type="bar", show=False, ax=ax_summary)
        else:
            shap.summary_plot(shap_values_summary, sample_df, plot_type="bar", show=False, ax=ax_summary)

        st.pyplot(fig_summary, use_container_width=True)
        plt.clf()
        plt.close(fig_summary)


    except Exception as e:
        st.error(f"An error occurred during prediction or SHAP explanation: {e}")
        st.info("Please check your input values and ensure your model and SHAP background data are compatible. "
                "Also, verify that the feature names used for input match the model's expected features.")

st.markdown("---")
st.markdown("Developed by Srikanth Nethikar")
