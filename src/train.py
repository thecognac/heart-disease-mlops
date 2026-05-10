# Import pandas for data handling
import pandas as pd

# Import joblib to save trained model
import joblib

# Import MLflow for experiment tracking
import mlflow

# Import train-test split function
from sklearn.model_selection import train_test_split

# Import machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import accuracy metric
from sklearn.metrics import accuracy_score


# -------------------------------
# STEP 1: Define Column Names
# -------------------------------
# UCI dataset does not contain column headers,
# so we manually define them.

columns = [
    "age",         # Age of patient
    "sex",         # Gender
    "cp",          # Chest pain type
    "trestbps",    # Resting blood pressure
    "chol",        # Cholesterol level
    "fbs",         # Fasting blood sugar
    "restecg",     # Resting ECG result
    "thalach",     # Maximum heart rate achieved
    "exang",       # Exercise induced angina
    "oldpeak",     # ST depression
    "slope",       # Slope of ST segment
    "ca",          # Number of major vessels
    "thal",        # Thalassemia
    "target"       # Heart disease target
]


# -------------------------------
# STEP 2: Load Dataset
# -------------------------------
# Read CSV file from data/raw folder

df = pd.read_csv(
    "data/raw/heart.csv",
    names=columns
)


# -------------------------------
# STEP 3: Handle Missing Values
# -------------------------------
# Dataset contains '?' symbols.
# Replace them with NA values.

df = df.replace("?", pd.NA)

# Remove rows containing missing values
df = df.dropna()


# -------------------------------
# STEP 4: Convert Data Types
# -------------------------------
# Convert all columns into float type

df = df.astype(float)


# -------------------------------
# STEP 5: Convert Target Variable
# -------------------------------
# Original dataset target values:
#
# 0 = No disease
# 1,2,3,4 = Disease present
#
# Convert into binary classification:
#
# 0 -> 0
# 1,2,3,4 -> 1

df["target"] = df["target"].apply(
    lambda x: 1 if x > 0 else 0
)


# -------------------------------
# STEP 6: Separate Features & Target
# -------------------------------

# X contains input features
X = df.drop("target", axis=1)

# y contains target labels
y = df["target"]


# -------------------------------
# STEP 7: Split Dataset
# -------------------------------
# 80% data for training
# 20% data for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# STEP 8: Define Models
# -------------------------------
# We use 2 models:
#
# 1. Random Forest
# 2. Logistic Regression

models = {

    "RandomForest": RandomForestClassifier(),

}


# Variables to store best model info
best_model = None
best_acc = 0


# -------------------------------
# STEP 9: Train Models
# -------------------------------

for name, model in models.items():

    # Start MLflow experiment run
    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(X_train, y_train)

        # Predict on test data
        preds = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, preds)

        # Log accuracy into MLflow
        mlflow.log_metric("accuracy", acc)

        # Print model accuracy
        print(name, "Accuracy:", acc)

        # Save best model
        if acc > best_acc:

            best_acc = acc

            best_model = model


# -------------------------------
# STEP 10: Save Best Model
# -------------------------------
# Save model inside models folder

joblib.dump(
    best_model,
    "models/model.pkl"
)

print("Best model saved successfully.")