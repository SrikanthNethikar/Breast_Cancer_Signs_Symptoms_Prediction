Early Breast Cancer Risk Prediction App
🌟 Overview

This interactive web application provides an early breast cancer risk prediction based on a comprehensive set of clinical, radiological, and lifestyle factors. Built with Streamlit, it offers an intuitive user interface for data input and leverages a machine learning model to provide risk assessments. A key feature of this app is its ability to explain the model's predictions using SHAP (SHapley Additive exPlanations), offering transparency and insights into feature importance.

Goal: To provide a user-friendly tool for preliminary risk assessment, aiding in early detection discussions.
🚀 Live Demo

Experience the app live here:

👉 Launch the Early Breast Cancer Risk Prediction App 👈

(Remember to replace https://your-app-name.streamlit.app with your actual deployed Streamlit app URL!)
✨ Features

    Interactive Input: User-friendly sliders and select boxes for entering various patient parameters across mean measurements, error metrics, worst measurements, and patient history/lifestyle.

    Risk Prediction: Provides a clear "High Risk" or "Low Risk" assessment based on the input data.

    SHAP Explanations:

        Waterfall Plot: Visualizes how each feature contributes to the individual prediction, pushing the output from the base value.

        Summary Bar Plot: Displays overall feature importance derived from the training dataset.

        CSV Report: Allows users to download a detailed report of input values and their corresponding SHAP values.

        Interactive Force Plot: (If successfully implemented and working) Provides a dynamic visualization of feature contributions.

🛠️ Technologies Used

    Python: The core programming language.

    Streamlit: For building the interactive web application.

    Scikit-learn: For the machine learning model (e.g., Random Forest Classifier, Gradient Boosting Classifier).

    Pandas: For data manipulation and handling.

    NumPy: For numerical operations.

    SHAP (SHapley Additive exPlanations): For model interpretability and explanation.

    Matplotlib: For rendering static plots within Streamlit.

    Joblib: For efficient saving and loading of the trained model and feature columns.

📁 Project Structure

Breast_Cancer_Early_Signs_Symptoms_Prediction/
├── app.py                     # Main Streamlit application file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files/folders to ignore in Git
├── model/                     # Contains trained ML models and related assets
│   ├── breast_cancer_risk_model.pkl # The trained machine learning model
│   └── model_columns.pkl      # List of feature names in correct order
├── data/                      # Contains datasets
│   └── breast_cancer_early_risk.csv # Dataset used for SHAP background data
│   └── shap_background_data.pkl # (Optional) Pre-processed SHAP background data
└── shap_summary_bar.png       # Static image for SHAP summary plot
└── ... (other files)

⚙️ How to Run Locally

To run this application on your local machine, follow these steps:

    Clone the repository:

    git clone https://github.com/SrikanthNethikar/Breast_Cancer_Signs_Symptoms_Prediction.git
    cd Breast_Cancer_Early_Signs_Symptoms_Prediction

    Create and activate a virtual environment (recommended):

    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    Install the required Python packages:

    pip install -r requirements.txt

    Run the Streamlit app:

    streamlit run app.py

    The app will open in your default web browser.

📊 Model Information

The core of this application is a machine learning model (e.g., a Gradient Boosting Classifier or Random Forest) trained on a dataset containing various clinical, radiological, and lifestyle features. The model_columns.pkl file ensures that input data is correctly aligned with the features the model expects.
💡 Interpretability with SHAP

Understanding why a model makes a certain prediction is crucial, especially in healthcare. This app integrates SHAP to provide clear, local explanations for each prediction. The waterfall plot shows how individual feature values influence the prediction for a specific patient, while the summary plot highlights the most impactful features across the entire dataset.
👤 Author

    Srikanth Nethikar

        Your LinkedIn Profile Link Here (Optional)

📄 License

This project is licensed under the MIT License.
