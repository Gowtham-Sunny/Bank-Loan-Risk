"""
Give Me Some Credit — Streamlit Dashboard
==========================================
Loads the LightGBM model + feature pipeline produced by
`give_me_some_credit_lgbm.ipynb` and exposes:

  * Overview        — dataset summary, class balance, key EDA charts
  * Score Applicant  — score a single applicant from manual inputs
  * Batch Scoring     — upload a CSV and score every row, download results
  * Model Insights    — validation metrics, ROC curve, feature importance

Run with:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
import feature_engineering as fe

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="Give Me Some Credit — Risk Dashboard",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "lgbm_model.txt"
    if not model_path.exists():
        return None
    return lgb.Booster(model_file=str(model_path))


@st.cache_resource
def load_artifacts():
    stats_path = MODEL_DIR / "feature_stats.json"
    metrics_path = MODEL_DIR / "metrics.json"
    if not stats_path.exists():
        return None, None
    with open(stats_path) as f:
        stats = json.load(f)
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return stats, metrics


@st.cache_data
def load_training_data():
    path = DATA_DIR / "cs-training.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def score_dataframe(raw_df: pd.DataFrame, model, stats):
    """Run the full feature-engineering + model pipeline on raw applicant rows."""
    engineered, _, _ = fe.engineer_features(
        raw_df,
        stats=stats["column_means"],
        train_mean_age=stats["train_mean_age"],
        is_training=False,
    )
    selected = stats["selected_features"]
    missing = [c for c in selected if c not in engineered.columns]
    for c in missing:
        engineered[c] = np.nan
    X = engineered[selected]
    return model.predict(X)


RAW_INPUT_COLUMNS = [c for c in fe.BASE_RAW_COLUMNS if c not in ("age",)]  # age handled separately

model = load_model()
stats, metrics = load_artifacts()
train_data = load_training_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("💳 Give Me Some Credit")
st.sidebar.caption("Credit default risk scoring dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Score an Applicant", "Batch Scoring", "Model Insights"],
)

if model is None or stats is None:
    st.sidebar.error(
        "Model artifacts not found in `model/`. Run "
        "`give_me_some_credit_lgbm.ipynb` first to train and save the model."
    )

if metrics:
    st.sidebar.metric("Validation AUC", f"{metrics.get('validation_auc', 0):.4f}")
    st.sidebar.metric("Features used", metrics.get("n_features_selected", "-"))
    st.sidebar.metric("Training rows", f"{metrics.get('n_train_rows', 0):,}")

st.sidebar.divider()
st.sidebar.caption(
    "Model: two-stage LightGBM (full feature sweep → top-100 features), "
    "ported from the original top-1 Kaggle R notebook."
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Give Me Some Credit — Overview")
    st.write(
        "This dataset asks: given an applicant's credit history and finances, "
        "what is the probability they experience serious financial distress "
        "(90+ days delinquent) within the next two years?"
    )

    if train_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Applicants", f"{len(train_data):,}")
        col2.metric("Default rate", f"{train_data['SeriousDlqin2yrs'].mean()*100:.2f}%")
        col3.metric("Median age", int(train_data["age"].median()))
        col4.metric("Missing income %", f"{train_data['MonthlyIncome'].isna().mean()*100:.1f}%")

        st.subheader("Class balance")
        counts = train_data["SeriousDlqin2yrs"].value_counts().rename({0: "No default", 1: "Default"})
        fig = px.bar(
            counts, x=counts.index, y=counts.values,
            labels={"x": "Outcome", "y": "Count"},
            color=counts.index, color_discrete_sequence=["#4C72B0", "#C44E52"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature distributions")
        col_a, col_b = st.columns(2)
        with col_a:
            feat = st.selectbox(
                "Choose a feature to inspect",
                ["age", "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines",
                 "NumberOfOpenCreditLinesAndLoans", "NumberOfDependents"],
            )
        clipped = train_data[feat].dropna()
        upper = clipped.quantile(0.99)
        fig2 = px.histogram(
            clipped[clipped <= upper], nbins=40,
            labels={"value": feat}, color_discrete_sequence=["#55A868"],
        )
        fig2.update_layout(showlegend=False, yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Default rate by age group")
        age_bins = pd.cut(train_data["age"], bins=[0, 25, 35, 45, 55, 65, 75, 110])
        rate_by_age = train_data.groupby(age_bins, observed=True)["SeriousDlqin2yrs"].mean()
        fig3 = px.bar(
            x=rate_by_age.index.astype(str), y=rate_by_age.values * 100,
            labels={"x": "Age group", "y": "Default rate (%)"},
            color_discrete_sequence=["#DD8452"],
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("`data/cs-training.csv` not found — EDA charts unavailable.")

# ---------------------------------------------------------------------------
# Page: Score an Applicant
# ---------------------------------------------------------------------------

elif page == "Score an Applicant":
    st.title("Score a Single Applicant")
    st.write("Enter applicant details to estimate their 2-year probability of serious delinquency.")

    if model is None:
        st.error("Model not loaded — train the notebook first.")
    else:
        with st.form("applicant_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=110, value=45)
                monthly_income = st.number_input("Monthly income ($)", min_value=0, value=5000, step=100)
                dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=0)
                revolving_util = st.number_input(
                    "Revolving utilization of unsecured lines (0-2, e.g. 0.3 = 30%)",
                    min_value=0.0, max_value=2.0, value=0.3, step=0.01, format="%.2f",
                )
            with c2:
                debt_ratio = st.number_input(
                    "Debt ratio (monthly debt / income)", min_value=0.0, value=0.3, step=0.01, format="%.2f"
                )
                open_credit_lines = st.number_input("Open credit lines & loans", min_value=0, value=6)
                real_estate_loans = st.number_input("Real estate loans/lines", min_value=0, value=1)
            with c3:
                late_30_59 = st.number_input("Times 30-59 days past due", min_value=0, value=0)
                late_60_89 = st.number_input("Times 60-89 days past due", min_value=0, value=0)
                late_90 = st.number_input("Times 90+ days late", min_value=0, value=0)

            submitted = st.form_submit_button("Score applicant", type="primary")

        if submitted:
            row = pd.DataFrame([{
                "RevolvingUtilizationOfUnsecuredLines": revolving_util,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": real_estate_loans,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfDependents": dependents,
            }])

            prob = score_dataframe(row, model, stats)[0]

            st.divider()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Predicted 2-year default probability", f"{prob*100:.2f}%")
                if prob >= 0.5:
                    st.error("High risk")
                elif prob >= (metrics.get("best_f1_threshold", 0.2) if metrics else 0.2):
                    st.warning("Elevated risk")
                else:
                    st.success("Low risk")
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#4C72B0"},
                        "steps": [
                            {"range": [0, 20], "color": "#dff0d8"},
                            {"range": [20, 50], "color": "#fcf8e3"},
                            {"range": [50, 100], "color": "#f2dede"},
                        ],
                    },
                    title={"text": "Default probability"},
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Batch Scoring
# ---------------------------------------------------------------------------

elif page == "Batch Scoring":
    st.title("Batch Scoring")
    st.write(
        "Upload a CSV with the same raw columns as `cs-training.csv` "
        "(the label column `SeriousDlqin2yrs` is optional and ignored if present). "
        "Or use the bundled Kaggle test file below."
    )

    use_sample = st.checkbox("Use bundled `data/cs-test.csv` instead of uploading", value=False)
    uploaded = None if use_sample else st.file_uploader("Upload CSV", type=["csv"])

    df_to_score = None
    if use_sample:
        sample_path = DATA_DIR / "cs-test.csv"
        if sample_path.exists():
            df_to_score = pd.read_csv(sample_path, index_col=0)
        else:
            st.error("`data/cs-test.csv` not found.")
    elif uploaded is not None:
        df_to_score = pd.read_csv(uploaded, index_col=0)

    if df_to_score is not None and model is not None:
        st.write(f"Loaded {len(df_to_score):,} rows.")
        raw_cols = df_to_score.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
        missing_cols = [c for c in fe.BASE_RAW_COLUMNS if c not in raw_cols.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
        else:
            with st.spinner("Scoring..."):
                probs = score_dataframe(raw_cols, model, stats)
            result = pd.DataFrame({"Id": df_to_score.index, "Probability": probs})

            st.subheader("Results preview")
            st.dataframe(result.head(20), use_container_width=True)

            st.subheader("Score distribution")
            fig = px.histogram(result, x="Probability", nbins=50, color_discrete_sequence=["#4C72B0"])
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )

# ---------------------------------------------------------------------------
# Page: Model Insights
# ---------------------------------------------------------------------------

elif page == "Model Insights":
    st.title("Model Insights")

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Validation AUC", f"{metrics.get('validation_auc', 0):.4f}")
        col2.metric("Features (full sweep)", metrics.get("n_features_full", "-"))
        col3.metric("Features (selected)", metrics.get("n_features_selected", "-"))

    st.subheader("Feature importance (final model)")
    if model is not None:
        imp = pd.DataFrame({
            "feature": model.feature_name(),
            "gain": model.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False).head(25)
        fig = px.bar(
            imp.sort_values("gain"), x="gain", y="feature", orientation="h",
            color_discrete_sequence=["#55A868"],
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model not loaded.")

    st.subheader("Validation ROC curve")
    if model is not None and train_data is not None and stats is not None:
        with st.spinner("Recomputing validation ROC..."):
            from sklearn.model_selection import train_test_split
            _, valid_df = train_test_split(
                train_data, test_size=0.15, random_state=42, stratify=train_data["SeriousDlqin2yrs"]
            )
            probs = score_dataframe(valid_df.drop(columns=["SeriousDlqin2yrs"]), model, stats)
            fpr, tpr, _ = roc_curve(valid_df["SeriousDlqin2yrs"], probs)
            auc = roc_auc_score(valid_df["SeriousDlqin2yrs"], probs)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"LightGBM (AUC={auc:.4f})", line=dict(color="#4C72B0", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("About the model")
    st.markdown(
        """
        * **Approach**: two-stage LightGBM, ported from the original top-1
          Kaggle R notebook.
        * **Stage 1**: trains on ~300+ engineered features (per-column
          transforms + all pairwise combinations of the 12 base variables)
          purely to rank feature importance.
        * **Stage 2**: retrains on the top 100 features by gain, with
          bagging and stronger regularization — this is the production
          model.
        * **Missing values**: handled natively by LightGBM; no imputation
          is performed.
        """
    )
