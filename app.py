import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* Main application background and default text */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0b1220 !important;
    color: #f1f5f9 !important;
}
[data-testid="stHeader"] {
    background-color: #0b1220 !important;
}
[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] span,
[data-testid="stMain"] li,
[data-testid="stMain"] small,
[data-testid="stMain"] .stMarkdown {
    color: #e2e8f0 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}

/* Sidebar: make navigation and labels clearly readable */
[data-testid="stSidebar"] {
    background-color: #111c2e !important;
}
[data-testid="stSidebar"] * {
    color: #e5edf8 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: transparent !important;
    color: #e5edf8 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
}

/* Metrics and cards */
[data-testid="stMetric"] {
    background: #172438 !important;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #334155;
}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #f8fafc !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #172438 !important;
    color: #f8fafc !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] * {
    color: #f8fafc !important;
}
[data-testid="stForm"] {
    background: #111c2e !important;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
}

/* Buttons, alerts, data tables */
.stButton > button, .stFormSubmitButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 0;
    border-radius: 8px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_files():
    return joblib.load("best_customer_churn_model.pkl"), joblib.load("feature_reference.pkl")

@st.cache_data
def load_dashboard_data():
    try:
        return pd.read_csv("dashboard_data.csv")
    except Exception:
        return pd.DataFrame()
