import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Generate synthetic dataset
n_samples = 1000
data = {
    "Employee_ID": range(1, n_samples + 1),
    "Age": np.random.randint(20, 60, n_samples),
    "Experience_Years": np.random.randint(1, 30, n_samples),
    "Monthly_Salary": np.random.randint(30000, 200000, n_samples),
    "Department": np.random.choice(["HR", "Engineering", "Sales", "Marketing"], n_samples),
    "Job_Role": np.random.choice(["Manager", "Engineer", "Analyst", "Consultant"], n_samples),
    "Work_Location": np.random.choice(["Office", "Remote", "Hybrid"], n_samples),
    "Performance_Rating": np.random.choice(["High", "Medium", "Low"], n_samples),
    "Promotion_Status": np.random.choice(["Yes", "No"], n_samples)
}

df = pd.DataFrame(data)

# Introduce missing values and duplicates to mimic inconsistencies
for col in ["Age", "Monthly_Salary"]:
    df.loc[np.random.choice(df.index, size=int(n_samples * 0.1), replace=False), col] = np.nan
df = pd.concat([df, df.sample(50, random_state=42)])  # Add duplicates

# Drop rows with missing target values
df = df.dropna(subset=["Promotion_Status"])

# Encode the target variable
df["Promotion_Status"] = df["Promotion_Status"].map({"Yes": 1, "No": 0})

# Separate features and target
X = df.drop(columns=["Employee_ID", "Promotion_Status"])
y = df["Promotion_Status"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)

# Feature selection and dimensionality reduction (only for numerical data)
feature_reduction = Pipeline(steps=[
    ("select_k_best", SelectKBest(score_func=f_classif, k=10)),
    ("pca", PCA(n_components=5))
])

# Create a pipeline that includes preprocessing, feature reduction, and the model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_reduction", feature_reduction),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=10,
        random_state=42))
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on raw data
raw_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=10,
        random_state=42))
])
raw_pipeline.fit(X_train, y_train)

# Evaluate the model on training data (raw)
y_train_pred_raw = raw_pipeline.predict(X_train)
train_accuracy_raw = accuracy_score(y_train, y_train_pred_raw)

# Train the model on preprocessed data
model.fit(X_train, y_train)

# Evaluate the model on training data (processed)
y_train_pred = model.predict(X_train)
train_accuracy_processed = accuracy_score(y_train, y_train_pred)

# Generate predictions on test data (raw)
y_test_pred_raw = raw_pipeline.predict(X_test)

# Generate predictions on test data (processed)
y_test_pred = model.predict(X_test)

# Visualizations
st.title('Employee Performance Analysis')

# 1. Data Distribution (Before and After Preprocessing)
st.subheader("1. Distribution of Age (Before Preprocessing)")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.histplot(df["Age"], kde=True, bins=20, color="skyblue", ax=ax1)
ax1.set_title("Distribution of Age (Before Preprocessing)")
ax1.set_xlabel("Age")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# 2. Model Performance Metrics (Comparison)
st.subheader("2. Model Performance (Accuracy Comparison)")
st.write(f"Training Accuracy (Raw Data): {train_accuracy_raw:.2f}")
st.write(f"Training Accuracy (Processed Data): {train_accuracy_processed:.2f}")

# 3. Other Relevant Visualizations
st.subheader("3. Correlation Heatmap of Numerical Features")
fig2, ax2 = plt.subplots(figsize=(10, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax2)
ax2.set_title("Correlation Heatmap of Numerical Features")
st.pyplot(fig2)

# 4. Classification Report
st.subheader("4. Classification Report (Raw Data)")
st.text(classification_report(y_test, y_test_pred_raw))

st.subheader("5. Classification Report (Processed Data)")
st.text(classification_report(y_test, y_test_pred))

# 5. Save Preprocessed Data
st.subheader("6. Save Preprocessed Data")
st.write("Click below to download the preprocessed data.")

# Provide download links for the preprocessed dataset
X_preprocessed = pd.DataFrame(
    model.named_steps["preprocessor"].transform(X),
    columns=numerical_columns.tolist() + list(
        model.named_steps["preprocessor"].transformers_[1][1]["onehot"].get_feature_names_out(categorical_columns))
)
X_preprocessed.columns = [f"Feature_{i}" if col == "" else col for i, col in enumerate(X_preprocessed.columns)]
X_preprocessed["Promotion_Status"] = y.reset_index(drop=True)

# Save to CSV
X_preprocessed.to_csv("employee_performance_data_preprocessed.csv", index=False)

# Provide download button for the preprocessed CSV
st.download_button(
    label="Download Preprocessed Data",
    data=X_preprocessed.to_csv(index=False),
    file_name="employee_performance_data_preprocessed.csv",
    mime="text/csv"
)

# Provide download button for the raw data
df.to_csv("employee_performance_data.csv", index=False)
st.download_button(
    label="Download Raw Data",
    data=df.to_csv(index=False),
    file_name="employee_performance_data.csv",
    mime="text/csv"
)

