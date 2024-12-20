import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Load datasets
df_raw = pd.read_csv("https://raw.githubusercontent.com/abhi-9876/VideoAssignment/refs/heads/main/employee_performance_data.csv")
df_preprocessed = pd.read_csv("https://raw.githubusercontent.com/abhi-9876/VideoAssignment/refs/heads/main/employee_performance_data_preprocessed.csv")

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Performance Dashboard", layout="wide")

# Title and description
st.title("Interactive Dashboard: Employee Performance Analysis")
st.markdown("""
This dashboard provides visualizations for:
- Data distribution before and after preprocessing
- Model performance metrics
- Other relevant insights
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
view = st.sidebar.selectbox(
    "Choose a view:",
    ["Raw Data Distribution", "Preprocessed Data Distribution", "Model Performance Metrics"]
)

# Raw data distribution
if view == "Raw Data Distribution":
    st.header("Data Distribution (Raw)")
    
    # Show raw dataset
    if st.checkbox("Show Raw Dataset"):
        st.write(df_raw.head())

    # Distribution of numerical columns
    numerical_columns = df_raw.select_dtypes(include=[np.number]).columns
    column = st.selectbox("Select a column to visualize:", numerical_columns)
    
    # Histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df_raw[column].dropna(), kde=True, bins=20, color="skyblue", ax=ax)
    ax.set_title(f"Distribution of {column} (Raw Data)")
    st.pyplot(fig)

# Preprocessed data distribution
elif view == "Preprocessed Data Distribution":
    st.header("Data Distribution (Preprocessed)")
    
    # Show preprocessed dataset
    if st.checkbox("Show Preprocessed Dataset"):
        st.write(df_preprocessed.head())

    # Distribution of numerical columns
    numerical_columns = df_preprocessed.select_dtypes(include=[np.number]).columns
    column = st.selectbox("Select a column to visualize:", numerical_columns)
    
    # Histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df_preprocessed[column].dropna(), kde=True, bins=20, color="green", ax=ax)
    ax.set_title(f"Distribution of {column} (Preprocessed Data)")
    st.pyplot(fig)

# Model performance metrics
elif view == "Model Performance Metrics":
    st.header("Model Performance Metrics")
    
    # Enter classification report
    raw_metrics = classification_report(
        [1, 0, 1, 0],  # Example true values
        [1, 0, 0, 0],  # Example predicted values for raw data
        output_dict=True
    )
    processed_metrics = classification_report(
        [1, 0, 1, 0],  # Example true values
        [1, 1, 1, 0],  # Example predicted values for processed data
        output_dict=True
    )

    st.subheader("Raw Data Metrics")
    st.table(pd.DataFrame(raw_metrics).transpose())

    st.subheader("Processed Data Metrics")
    st.table(pd.DataFrame(processed_metrics).transpose())

    # Comparison of accuracies
    accuracies = {
        "Raw Data": 0.75,  # Example accuracy for raw data
        "Processed Data": 0.85  # Example accuracy for processed data
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="Set2", ax=ax)
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Footer
st.sidebar.markdown("""---
**Created by Abhijith S Menon**
""")
