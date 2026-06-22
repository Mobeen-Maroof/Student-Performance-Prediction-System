import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ====================================
# LOAD DATASET
# ====================================

df = pd.read_csv("student_performance_prediction.csv")

print("\nFirst 5 Rows:\n")
print(df.head())

# ====================================
# REMOVE DUPLICATES
# ====================================

df.drop_duplicates(inplace=True)

# ====================================
# REMOVE EMPTY TARGET VALUES
# ====================================

df = df.dropna(subset=["Passed"])

print("\nDataset Shape:", df.shape)

# ====================================
# FEATURES AND TARGET
# ====================================

X = df.drop(columns=["Student ID", "Passed"])

y = df["Passed"]

# ====================================
# NUMERIC AND CATEGORICAL COLUMNS
# ====================================

numeric_features = [
    "Study Hours per Week",
    "Attendance Rate",
    "Previous Grades"
]

categorical_features = [
    "Participation in Extracurricular Activities",
    "Parent Education Level"
]

# ====================================
# NUMERIC PIPELINE
# ====================================

numeric_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="mean")),

    ("scaler", StandardScaler())

])

# ====================================
# CATEGORICAL PIPELINE
# ====================================

categorical_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("encoder", OneHotEncoder(handle_unknown="ignore"))

])

# ====================================
# COMBINE PREPROCESSING
# ====================================

preprocessor = ColumnTransformer(

    transformers=[

        ("num", numeric_transformer, numeric_features),

        ("cat", categorical_transformer, categorical_features)

    ]

)

# ====================================
# TRAIN TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ====================================
# MODEL PIPELINE
# ====================================

model_pipeline = Pipeline(steps=[

    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(

        n_estimators=200,
        max_depth=10,
        random_state=42

    ))

])

# ====================================
# TRAIN MODEL
# ====================================

print("\nTraining Model...\n")

model_pipeline.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# ====================================
# PREDICTIONS
# ====================================

y_pred = model_pipeline.predict(X_test)

# ====================================
# MODEL ACCURACY
# ====================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ====================================
# CLASSIFICATION REPORT
# ====================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ====================================
# CONFUSION MATRIX
# ====================================

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# ====================================
# SAVE MODEL
# ====================================

joblib.dump(model_pipeline, "student_model.pkl")

print("\nModel Saved Successfully as student_model.pkl")