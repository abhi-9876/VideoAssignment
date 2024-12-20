import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report

# Load datasets from URLs
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
    
    # Create histogram for raw data using plotly.graph_objects
    hist_data = df_raw[column].dropna()
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=hist_data, nbinsx=20, name=f"Distribution of {column} (Raw Data)"))
    hist_fig.update_layout(title=f"Distribution of {column} (Raw Data)", xaxis_title=column, yaxis_title="Frequency")
    st.plotly_chart(hist_fig)

# Preprocessed data distribution
elif view == "Preprocessed Data Distribution":
    st.header("Data Distribution (Preprocessed)")
    
    # Show preprocessed dataset
    if st.checkbox("Show Preprocessed Dataset"):
        st.write(df_preprocessed.head())

    # Distribution of numerical columns
    numerical_columns = df_preprocessed.select_dtypes(include=[np.number]).columns
    column = st.selectbox("Select a column to visualize:", numerical_columns)
    
    # Create histogram for preprocessed data using plotly.graph_objects
    hist_data = df_preprocessed[column].dropna()
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=hist_data, nbinsx=20, name=f"Distribution of {column} (Preprocessed Data)"))
    hist_fig.update_layout(title=f"Distribution of {column} (Preprocessed Data)", xaxis_title=column, yaxis_title="Frequency")
    st.plotly_chart(hist_fig)

# Model performance metrics
elif view == "Model Performance Metrics":
    st.header("Model Performance Metrics")
    
    # Example classification reports (replace with actual model predictions)
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

    # Accuracy comparison
    accuracies = {
        "Raw Data": 0.75,  # Example accuracy for raw data
        "Processed Data": 0.85  # Example accuracy for processed data
    }

    # Create bar plot for accuracy comparison using plotly.graph_objects
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
        name="Accuracy Comparison",
        marker=dict(color='rgba(55, 128, 191, 0.7)'),
    ))
    bar_fig.update_layout(title="Accuracy Comparison", xaxis_title="Data Type", yaxis_title="Accuracy")
    st.plotly_chart(bar_fig)

# Footer
st.sidebar.markdown("""---
**Created by Abhijith S Menon**
""")
