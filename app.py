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

try:
    model, ref = load_model_files()
except Exception as e:
    st.error("Model files could not be loaded. Check that the model and feature_reference files are in the repository.")
    st.exception(e)
    st.stop()

data = load_dashboard_data()
features = ref["feature_columns"]
cat_cols = ref["categorical_columns"]
num_cols = ref["numeric_columns"]
category_options = ref["category_options"]
numeric_defaults = ref["numeric_defaults"]
numeric_dtypes = ref.get("numeric_dtypes", {})

def get_risk(p):
    return "Low Risk" if p < 0.30 else ("Medium Risk" if p < 0.60 else "High Risk")

def make_recommendations(customer, risk):
    recs = []
    if customer.get("Contract") == "Month-to-month":
        recs.append("Offer an incentive to move to a longer-term contract.")
    if customer.get("tenure", 99) < 12:
        recs.append("Provide onboarding support and proactive early engagement.")
    if customer.get("TechSupport") == "No":
        recs.append("Offer suitable technical support assistance.")
    if customer.get("OnlineSecurity") == "No":
        recs.append("Explain available online security services.")
    if customer.get("PaymentMethod") == "Electronic check":
        recs.append("Encourage a convenient automatic payment method.")
    if customer.get("MonthlyCharges", 0) > 70:
        recs.append("Review plan suitability and possible pricing options.")
    if risk == "High Risk":
        recs.insert(0, "Prioritize this customer for a retention-team follow-up.")
    elif risk == "Medium Risk":
        recs.insert(0, "Schedule a proactive customer satisfaction check-in.")
    return recs or ["Continue regular engagement and monitor customer satisfaction."]

st.sidebar.title("📊 Churn Analytics")
st.sidebar.caption("Machine Learning & Explainable AI")
page = st.sidebar.radio("NAVIGATION", [
    "Dashboard", "Churn Analysis", "Customer Prediction",
    "Model Comparison", "Feature Importance", "Retention Recommendations",
    "Historical Records"
])
st.sidebar.divider()
st.sidebar.success(f"Model loaded: {ref.get('best_model_name', 'Selected model')}")

st.title("Customer Churn Analytics")
st.caption("Predict churn • Understand risk • Recommend retention actions")

if page == "Dashboard":
    st.subheader("Business Overview")
    if not data.empty and "Churn" in data.columns:
        churn = pd.to_numeric(data["Churn"], errors="coerce")
        total = len(data)
        churned = int((churn == 1).sum())
        retained = int((churn == 0).sum())
        rate = 100 * churned / total if total else 0
    else:
        total = churned = retained = 0
        rate = 0
    a,b,c,d = st.columns(4)
    a.metric("Total customers", f"{total:,}" if total else "—")
    b.metric("Churned customers", f"{churned:,}" if total else "—")
    c.metric("Retained customers", f"{retained:,}" if total else "—")
    d.metric("Churn rate", f"{rate:.2f}%" if total else "—")
    if not data.empty and "Churn" in data.columns:
        left,right = st.columns(2)
        with left:
            st.markdown("#### Customer status")
            st.bar_chart(data["Churn"].map({0:"Retained",1:"Churned"}).value_counts())
        with right:
            st.markdown("#### Churn rate by contract")
            if "Contract" in data.columns:
                st.bar_chart(data.groupby("Contract")["Churn"].mean().mul(100))
        if "PaymentMethod" in data.columns:
            st.markdown("#### Churn rate by payment method")
            st.bar_chart(data.groupby("PaymentMethod")["Churn"].mean().mul(100))
    else:
        st.info("Add dashboard_data.csv to the repository to display historical dashboard statistics.")

elif page == "Churn Analysis":
    st.subheader("Explore churn patterns")
    if data.empty or "Churn" not in data.columns:
        st.warning("dashboard_data.csv is required for historical analysis.")
    else:
        filtered = data.copy()
        filter_cols = st.columns(3)
        for i,col in enumerate(["Contract","InternetService","PaymentMethod"]):
            if col in filtered.columns:
                opts = ["All"] + sorted(filtered[col].dropna().astype(str).unique())
                choice = filter_cols[i].selectbox(col, opts)
                if choice != "All":
                    filtered = filtered[filtered[col].astype(str) == choice]
        total = len(filtered)
