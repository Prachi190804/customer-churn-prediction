churned = int((filtered["Churn"] == 1).sum())
        x,y,z = st.columns(3)
        x.metric("Filtered customers", f"{total:,}")
        y.metric("Churned", f"{churned:,}")
        z.metric("Churn rate", f"{100*churned/total:.2f}%" if total else "0%")
        if total and "tenure" in filtered.columns:
            bins = pd.cut(filtered["tenure"], [-1,12,24,48,60,100],
                          labels=["0–12","13–24","25–48","49–60","60+"])
            st.markdown("#### Churn rate by tenure group")
            st.bar_chart(filtered.assign(TenureGroup=bins).groupby("TenureGroup", observed=False)["Churn"].mean().mul(100))
        st.dataframe(filtered.head(200), use_container_width=True)

elif page == "Customer Prediction":
    st.subheader("Individual customer prediction")
    customer = {}
    with st.form("customer_form"):
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                if feature in cat_cols:
                    opts = category_options.get(feature, [])
                    if opts:
                        customer[feature] = st.selectbox(feature, opts, key=f"cat_{feature}")
                    else:
                        customer[feature] = st.text_input(feature, key=f"cat_{feature}")
                else:
                    default = float(numeric_defaults.get(feature, 0))
                    val = st.number_input(feature, min_value=0.0, value=default, step=1.0, key=f"num_{feature}")
                    if numeric_dtypes.get(feature, "").startswith("int"):
                        val = int(val)
                    customer[feature] = val
        submitted = st.form_submit_button("Predict Customer Churn", use_container_width=True)
    if submitted:
        try:
            row = pd.DataFrame([customer], columns=features)
            probability = float(model.predict_proba(row)[0][1])
            prediction = int(probability >= 0.5)
            risk = get_risk(probability)
            st.divider()
            a,b,c = st.columns(3)
            a.metric("Churn probability", f"{probability:.1%}")
            b.metric("Prediction", "Likely to churn" if prediction else "Likely to stay")
            c.metric("Risk level", risk)
            st.progress(min(max(probability,0),1))
            if risk == "High Risk":
                st.error("High predicted churn risk. Prioritize retention follow-up.")
            elif risk == "Medium Risk":
                st.warning("Medium predicted churn risk. Proactive engagement is recommended.")
            else:
                st.success("Low predicted churn risk.")
            st.session_state["last_customer"] = customer
            st.session_state["last_risk"] = risk
            st.session_state["last_probability"] = probability
            st.markdown("#### Retention recommendations")
            for rec in make_recommendations(customer, risk):
                st.write("• " + rec)
        except Exception as e:
            st.error("Prediction failed. Confirm the saved model and feature reference were created by the same notebook run.")
            st.exception(e)

elif page == "Model Comparison":
    st.subheader("Model performance comparison")
    try:
        comparison = pd.read_csv("model_comparison_results.csv", index_col=0)
        st.dataframe(comparison.round(4), use_container_width=True)
        numeric = comparison.select_dtypes(include="number")
        if not numeric.empty:
            st.bar_chart(numeric)
    except Exception:
        st.info("Upload model_comparison_results.csv from Colab to show model metrics.")

elif page == "Feature Importance":
    st.subheader("Global model feature importance")
    try:
        fitted = model
        names = features
        if hasattr(model, "named_steps"):
            fitted = model.named_steps["model"]
            names = model.named_steps["prep"].get_feature_names_out()
        importance = getattr(fitted, "feature_importances_", None)
        if importance is None:
            st.info("Built-in feature importance is not available for this selected model.")
        else:
            imp = pd.DataFrame({"Feature": names, "Importance": importance}).sort_values("Importance", ascending=False).head(20)
            st.bar_chart(imp.set_index("Feature"))
            st.dataframe(imp, hide_index=True, use_container_width=True)
            st.caption("Global feature importance describes model-level patterns; it does not establish causation.")
    except Exception as e:
        st.warning("Could not display feature importance for this model.")
        st.exception(e)
    st.info("Your notebook's SHAP cells provide global and individual SHAP explanations. This page currently displays built-in model importance.")

elif page == "Retention Recommendations":
    st.subheader("Retention action guide")
    guide = pd.DataFrame([
        ["Month-to-month contract","Offer a longer-term contract incentive."],
        ["Short tenure","Provide onboarding and early customer support."],
        ["No technical support","Offer suitable technical support."],
        ["No online security","Explain available security services."],
        ["Electronic check","Promote convenient automatic payment options."],
        ["High monthly charges","Review plan suitability and pricing."]
    ], columns=["Risk signal","Suggested action"])
    st.dataframe(guide, hide_index=True, use_container_width=True)
    if "last_customer" in st.session_state:
        st.metric("Latest predicted churn probability", f"{st.session_state['last_probability']:.1%}")
        for rec in make_recommendations(st.session_state["last_customer"], st.session_state["last_risk"]):
            st.write("• " + rec)
    else:
        st.info("Run a customer prediction to see customer-specific recommendations.")

elif page == "Historical Records":
    st.subheader("Historical customer records")
    if data.empty:
        st.warning("dashboard_data.csv is required for historical records.")
    else:
        view = data.copy()
        if "Churn" in view.columns:
            status = st.selectbox("Customer status", ["All","Churned","Retained"])
            if status == "Churned":
                view = view[view["Churn"] == 1]
            elif status == "Retained":
                view = view[view["Churn"] == 0]
        st.caption(f"Showing {len(view):,} records")
        st.dataframe(view, use_container_width=True, height=450)
        st.download_button("Download filtered records", view.to_csv(index=False).encode("utf-8"),
                           "filtered_customer_records.csv", "text/csv")

st.sidebar.divider()
