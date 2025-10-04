import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
from datetime import datetime
import os

# --- File Paths ---
# DVC will manage the 'data' directory.
# This script expects the data to be at 'data/iris.csv'
data_path = os.path.join('data', 'iris.csv')
local_model_path = 'model.joblib'
local_log_path = 'metrics.txt'

# --- Load Dataset ---
# Added a check to ensure the data file exists.
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_path}'")
    print("Please make sure your data is present before running the script.")
    exit()

# --- Data Splitting ---
# Stratified split to maintain the same proportion of species in train and test sets.
train, test = train_test_split(data,
                             test_size=0.4,
                             stratify=data['species'],
                             random_state=42)

# --- Feature and Target Selection ---
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X_train = train[features]
y_train = train['species']
X_test = test[features]
y_test = test['species']

# --- Model Training ---
# Initialize and train the Decision Tree model as specified.
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)

# --- Prediction and Evaluation ---
prediction = mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction, y_test)

print(f'The accuracy of the Decision Tree is: {accuracy:.3f}')

# --- Save Outputs ---
# The model and metrics files will be versioned by DVC.

# Save the trained model
joblib.dump(mod_dt, local_model_path)
print(f"Model saved to '{local_model_path}'")

# Get the current timestamp for the log
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Write metrics and other info to a text file
with open(local_log_path, "w") as log_file:
    log_file.write(f"Training Timestamp: {timestamp}\n")
    log_file.write(f"Model: DecisionTreeClassifier(max_depth=3, random_state=1)\n")
    log_file.write(f"Accuracy: {accuracy:.4f}\n")
    log_file.write(f"Features used: {', '.join(features)}\n")

print(f"Metrics saved to '{local_log_path}'")
