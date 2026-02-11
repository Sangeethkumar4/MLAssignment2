"""
ML Classification Models - Streamlit Web Application
Dataset: Heart Disease UCI Dataset
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🫀",
    layout="wide"
)

# Title
st.title("🫀 ML Classification Models Comparison")
st.markdown("### Heart Disease UCI Dataset - Binary Classification")
st.markdown("---")

@st.cache_data
def load_default_data():
    """Load Heart Disease UCI dataset"""
    url = "https://raw.githubusercontent.com/rashidwassan/heart-disease-uci-dataset/main/heart_disease_uci.csv"
    df = pd.read_csv(url)

    # Drop id and dataset columns
    df = df.drop(['id', 'dataset'], axis=1)

    # Handle missing values
    df = df.dropna()

    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop('num', axis=1)

    # Encode categorical variables
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

def preprocess_uploaded_data(df):
    """Preprocess uploaded data"""
    # Drop id and dataset columns if they exist
    cols_to_drop = ['id', 'dataset']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Handle num column if exists
    if 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)
        df = df.drop('num', axis=1)

    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    return df

def get_model(model_name):
    """Return the selected model"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    return models[model_name]

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    if y_prob is not None:
        try:
            if len(y_prob.shape) > 1:
                metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0

    return metrics

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    return fig

# Sidebar
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (Test Data)", type=['csv'])

st.sidebar.header("🔧 Model Selection")
model_name = st.sidebar.selectbox(
    "Select Classification Model",
    ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost']
)

# Load data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_uploaded_data(data)
    st.sidebar.success("✅ File uploaded successfully!")
else:
    data = load_default_data()
    st.sidebar.info("ℹ️ Using default Heart Disease UCI dataset")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dataset Information")
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    st.write(f"**Features:** {data.shape[1] - 1}")
    st.write(f"**Target Distribution:**")
    target_dist = data['target'].value_counts()
    st.write(f"- No Disease (0): {target_dist.get(0, 0)}")
    st.write(f"- Disease (1): {target_dist.get(1, 0)}")

with col2:
    st.subheader("🔍 Data Preview")
    st.dataframe(data.head(), use_container_width=True)

st.markdown("---")

# Train and evaluate model
if st.button("🚀 Train and Evaluate Model", type="primary"):
    with st.spinner(f"Training {model_name}..."):
        # Prepare data
        X = data.drop('target', axis=1)
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = get_model(model_name)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)
        else:
            y_prob = None

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Display results
        st.success(f"✅ {model_name} trained successfully!")

        st.subheader(f"📈 Evaluation Metrics - {model_name}")

        # Metrics in columns
        metric_cols = st.columns(6)
        metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']

        for i, metric in enumerate(metric_names):
            with metric_cols[i]:
                st.metric(label=metric, value=f"{metrics[metric]:.4f}")

        st.markdown("---")

        # Confusion Matrix and Classification Report
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎯 Confusion Matrix")
            fig = plot_confusion_matrix(cm, model_name)
            st.pyplot(fig)

        with col2:
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# Compare all models section
st.markdown("---")
st.subheader("📊 Compare All Models")

if st.button("🔄 Run All Models Comparison", type="secondary"):
    with st.spinner("Training all models..."):
        X = data.drop('target', axis=1)
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        all_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        }

        results = []

        progress_bar = st.progress(0)
        for idx, (name, model) in enumerate(all_models.items()):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)
            else:
                y_prob = None

            metrics = calculate_metrics(y_test, y_pred, y_prob)
            metrics['Model'] = name
            results.append(metrics)

            progress_bar.progress((idx + 1) / len(all_models))

        results_df = pd.DataFrame(results)
        results_df = results_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']]

        st.success("✅ All models trained successfully!")
        st.dataframe(
            results_df.style.highlight_max(subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'], color='lightgreen'),
            use_container_width=True
        )

        # Bar chart comparison
        st.subheader("📊 Model Performance Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        results_df.set_index('Model')[['Accuracy', 'AUC', 'F1 Score', 'MCC']].plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Dataset:** Heart Disease UCI Dataset | **Features:** 13 | **Instances:** 920")
st.markdown("*Built with Streamlit for ML Assignment 2*")
