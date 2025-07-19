# model_training.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Added for saving the model

# 1. Load the dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names) # Features
y = pd.Series(cancer.target) # Target (0 for malignant, 1 for benign)

print("Dataset loaded. First 5 rows of features (X):")
print(X.head())
print("\nTarget values (y) distribution:")
print(y.value_counts()) # 0: Malignant, 1: Benign

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 3. Choose and train a model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=cancer.target_names)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save the trained model to a file (NEWLY ADDED)
model_filename = 'decision_tree_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved successfully as {model_filename}")