
import streamlit as st
import pandas as pd
import joblib

# Load saved model and feature information
model = joblib.load("best_customer_churn_model.pkl")
reference = joblib.load("feature_reference.pkl")

feature_columns = reference["feature_columns"]
categorical_columns = reference["categorical_columns"]
numeric_columns = reference["numeric_columns"]
category_options = reference["category_options"]
numeric_defaults = reference["numeric_defaults"]
model_name = reference["best_model_name"]


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
h1 {
    color: #17365d;
}
.stButton>button {
    background-color: #17365d;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# Header
st.title("Customer Churn Prediction")
st.subheader("Machine Learning and Explainable AI")
st.write(
    "Enter customer details to predict churn probability "
    "and receive a retention recommendation."
)

st.info(f"Prediction model: {model_name}")

# Customer input form
st.header("Customer Information")

customer_data = {}

with st.form("customer_form"):

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_columns):

        current_col = col1 if index % 2 == 0 else col2

        with current_col:

            if feature in categorical_columns:

                options = category_options[feature]

                customer_data[feature] = st.selectbox(
                    feature,
                    options=options
                )

            else:

                default = numeric_defaults[feature]

                customer_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=float(default),
                    step=1.0
                )

    submitted = st.form_submit_button("Predict Customer Churn")


# Prediction
if submitted:

    input_df = pd.DataFrame(
        [customer_data],
        columns=feature_columns
    )

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.divider()
    st.header("Churn Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    col2.metric(
        "Prediction",
        "Churn" if prediction else "No Churn"
    )

    col3.metric(
        "Risk Level",
        risk
    )

    st.progress(probability)

    # Retention recommendations
    st.header("Retention Recommendations")

    recommendations = []

    if customer_data.get("Contract") == "Month-to-month":
        recommendations.append(
            "Offer a discount or incentive for switching to a longer-term contract."
        )

    if customer_data.get("tenure", 0) < 12:
        recommendations.append(
            "Provide onboarding support and early-stage customer engagement."
        )

    if customer_data.get("TechSupport") == "No":
        recommendations.append(
            "Offer technical support assistance or a suitable support package."
        )

    if customer_data.get("OnlineSecurity") == "No":
        recommendations.append(
            "Promote an appropriate online security service."
        )

    if customer_data.get("PaymentMethod") == "Electronic check":
        recommendations.append(
            "Encourage convenient automatic payment options."
        )

    if customer_data.get("MonthlyCharges", 0) > 70:
        recommendations.append(
            "Review the customer's plan and consider a suitable pricing offer."
        )

    if risk == "High Risk":
        st.warning("This customer has a high predicted churn risk.")

    elif risk == "Medium Risk":
        st.info("This customer has a medium predicted churn risk.")

    else:
        st.success("This customer has a low predicted churn risk.")

    if recommendations:
        for recommendation in recommendations:
            st.write("•", recommendation)
    else:
        st.write(
            "Continue regular customer engagement and monitor satisfaction."
        )

    st.caption(
        "Note: Predictions are estimates based on historical customer data. "
        "Retention recommendations are rule-based suggestions."
    )
