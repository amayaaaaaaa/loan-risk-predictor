
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Home Credit Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

THRESHOLD = 0.50

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("HomeCreditModel.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load HomeCreditModel.joblib: {e}")
    st.stop()

# ---------------- BASELINE APPLICANT ----------------
def create_baseline():
    preprocessor = model.named_steps["preprocessing"]
    expected_features = list(preprocessor.feature_names_in_)

    # Object dtype allows both numeric and categorical values.
    applicant = pd.DataFrame(
        index=[0],
        columns=expected_features,
        dtype=object
    )

    # Fill numeric features using the fitted median imputers.
    num_pipeline = preprocessor.named_transformers_["num"]
    numeric_cols = list(preprocessor.transformers_[0][2])
    numeric_values = num_pipeline.named_steps["imputer"].statistics_

    for col, value in zip(numeric_cols, numeric_values):
        if col in applicant.columns:
            applicant.loc[0, col] = value

    # Fill categorical features using fitted most-frequent imputers.
    cat_pipeline = preprocessor.named_transformers_["cat"]
    categorical_cols = list(preprocessor.transformers_[1][2])
    categorical_values = cat_pipeline.named_steps["imputer"].statistics_

    for col, value in zip(categorical_cols, categorical_values):
        if col in applicant.columns:
            applicant.loc[0, col] = value

    return applicant

# ---------------- HEADER ----------------
st.title("🏦 Home Credit Default Risk Assessment")
st.write(
    "Enter applicant details to estimate the probability "
    "of repayment difficulties."
)

st.warning(
    "Educational demonstration only. This prediction is not "
    "a final loan approval or rejection decision."
)

st.info(
    "Decision threshold: 50%. A predicted probability of "
    "50% or higher is classified as At Risk."
)

# ---------------- INPUT FORM ----------------
st.header("Applicant Information")

left, right = st.columns(2)

with left:
    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=180000.0,
        step=10000.0
    )

    credit = st.number_input(
        "Credit Amount",
        min_value=0.0,
        value=450000.0,
        step=10000.0
    )

    annuity = st.number_input(
        "Loan Annuity (annual payment amount)",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    goods_price = st.number_input(
        "Price of Goods",
        min_value=0.0,
        value=400000.0,
        step=10000.0
    )

with right:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=1.0
    )

    family_members = st.number_input(
        "Number of Family Members",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    ext_score_2 = st.number_input(
        "External Credit Score 2 (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

# ---------------- PREDICTION ----------------
if st.button("Predict Repayment Risk", type="primary"):
    try:
        applicant = create_baseline()

        # Update only features confirmed to exist in the model.
        user_inputs = {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price,
            "DAYS_BIRTH": -int(age * 365.25),
            "DAYS_EMPLOYED": -int(years_employed * 365.25),
            "CNT_FAM_MEMBERS": family_members,
            "EXT_SOURCE_2": ext_score_2
        }

        for col, value in user_inputs.items():
            if col in applicant.columns:
                applicant.loc[0, col] = value
            else:
                st.error(
                    f"The trained model does not contain "
                    f"the required feature: {col}"
                )
                st.stop()

        # Predict default probability and apply the threshold.
        probability = float(model.predict_proba(applicant)[0, 1])
        predicted_class = int(probability >= THRESHOLD)

        st.divider()
        st.header("Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Risk Class",
            "At Risk" if predicted_class == 1
            else "Lower Predicted Risk"
        )

        col2.metric(
            "Probability of Repayment Difficulty",
            f"{probability:.2%}"
        )

        col3.metric("Decision Threshold", "50%")

        st.progress(float(np.clip(probability, 0.0, 1.0)))

        st.caption(
            "Eight features are customized. All remaining features "
            "use fitted imputation statistics, so this is an "
            "illustrative estimate, not a complete applicant-specific "
            "assessment."
        )

        # ---------------- OPTIONAL SHAP ----------------
        with st.expander("Why did the model make this prediction?"):
            st.write(
                "SHAP estimates how features influence model output. "
                "Feature influence does not establish causation."
            )

            try:
                preprocessor = model.named_steps["preprocessing"]
                rf_model = model.named_steps["model"]

                transformed = preprocessor.transform(applicant)
                feature_names = preprocessor.get_feature_names_out()

                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(transformed)

                if isinstance(shap_values, list):
                    values = shap_values[1][0]
                elif (
                    isinstance(shap_values, np.ndarray)
                    and shap_values.ndim == 3
                ):
                    values = shap_values[0, :, 1]
                else:
                    values = shap_values[0]

                expected = explainer.expected_value

                if isinstance(expected, (list, np.ndarray)):
                    base_value = np.asarray(expected).flatten()[-1]
                else:
                    base_value = expected

                feature_values = (
                    transformed.toarray()[0]
                    if hasattr(transformed, "toarray")
                    else np.asarray(transformed)[0]
                )

                explanation = shap.Explanation(
                    values=values,
                    base_values=base_value,
                    data=feature_values,
                    feature_names=feature_names
                )

                shap.plots.waterfall(
                    explanation,
                    max_display=12,
                    show=False
                )

                st.pyplot(plt.gcf(), clear_figure=True)
                plt.close("all")

            except Exception as e:
                st.error(f"Could not generate explanation: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Educational demonstration only. Do not use this model as "
    "the sole basis for real lending decisions."
)
