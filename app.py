import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Financial Deprivation & NHS Mental Health Service Demand",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Load everything once
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("models.pkl", "rb") as f:
        saved = pickle.load(f)
    features = pd.read_csv("model_features.csv")
    target = pd.read_csv("model_target.csv")["high_demand"]
    engineered = pd.read_csv("engineered.csv")
    shap_values = pd.read_csv("shap_values.csv")
    with open("shap_ranking.json") as f:
        shap_ranking = json.load(f)
    return saved, features, target, engineered, shap_values, shap_ranking

saved, features, target, engineered, shap_values, shap_ranking = load_artifacts()

MODEL_METRICS = {
    "Logistic Regression": {"AUC-ROC": 0.981, "Precision": 0.880, "Recall": 0.964, "F1": 0.920},
    "Random Forest":       {"AUC-ROC": 0.988, "Precision": 0.944, "Recall": 0.978, "F1": 0.961},
    "XGBoost":             {"AUC-ROC": 0.990, "Precision": 0.949, "Recall": 0.949, "F1": 0.949},
}

# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Model Comparison", "SHAP Predictor Ranking", "Explore a Demographic Factor", "About This Project"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "MSc Data Analytics with Banking and Finance dissertation project. "
    "Built on NHS England open data (2016–2023)."
)

# ============================================================
# Page: Overview
# ============================================================
if page == "Overview":
    st.title("Does Financial Deprivation Predict NHS Mental Health Service Demand?")
    st.markdown("##### A Machine Learning Analysis of NHS England Open Data")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows analysed", f"{len(engineered):,}")
    col2.metric("Best model (AUC-ROC)", "XGBoost — 0.990")
    col3.metric("Top SHAP predictor", "Accommodation")

    st.markdown("---")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What this project tests")
        st.write(
            "NHS mental health referrals have risen sharply since 2019, and financial deprivation "
            "is widely assumed to be a key driver of that demand. This project tests that assumption "
            "directly: three machine learning models were trained on NHS England's own service data "
            "across eight demographic breakdowns, and SHAP explainability was used to rank financial "
            "deprivation against every other available demographic factor, age, ethnicity, disability, "
            "employment status, sexual orientation, gender, and accommodation type, rather than "
            "assuming deprivation is automatically the strongest predictor."
        )
        st.subheader("The headline finding")
        st.write(
            "Deprivation did **not** come out on top. Accommodation type outranked it, suggesting that "
            "a narrower, more specific measure of hardship may carry a sharper signal than the "
            "Index of Multiple Deprivation's seven-domain composite score. Use the pages on the left "
            "to explore the model performance and the full SHAP ranking behind that finding."
        )
    with right:
        st.subheader("Breakdown sizes in this dataset")
        counts = engineered["BREAKDOWN"].str.replace("England; ", "", regex=False).value_counts()
        fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                     labels={"x": "Rows", "y": ""}, color_discrete_sequence=["#1B2A4A"])
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width='stretch')

# ============================================================
# Page: Model Comparison
# ============================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write(
        "Three models were trained and compared on an 80/20 stratified train-test split: "
        "Logistic Regression as an interpretable baseline, Random Forest as an ensemble comparison, "
        "and XGBoost as the strongest gradient-boosting candidate."
    )

    metrics_df = pd.DataFrame(MODEL_METRICS).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(
        metrics_df.style.format({"AUC-ROC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})
        .highlight_max(subset=["AUC-ROC", "Precision", "Recall", "F1"], color="#d4edda"),
        width='stretch', hide_index=True,
    )

    fig = go.Figure()
    for metric in ["AUC-ROC", "Precision", "Recall", "F1"]:
        fig.add_trace(go.Bar(name=metric, x=list(MODEL_METRICS.keys()),
                              y=[MODEL_METRICS[m][metric] for m in MODEL_METRICS]))
    fig.update_layout(barmode="group", yaxis_range=[0.8, 1.0], height=420,
                       margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width='stretch')

    st.info(
        "XGBoost was selected for the SHAP explainability stage, since it achieved the highest "
        "AUC-ROC of the three models. All scores are high partly because the dataset spans only "
        "twelve months, meaning specific categories are consistently high or low demand across that "
        "window, a limitation discussed further in the dissertation report."
    )

# ============================================================
# Page: SHAP Predictor Ranking
# ============================================================
elif page == "SHAP Predictor Ranking":
    st.title("SHAP Predictor Ranking")
    st.write(
        "SHAP (SHapley Additive exPlanations) was applied to the XGBoost model to rank each "
        "demographic breakdown by how strongly it drives the model's predictions, aggregating "
        "importance across every category within that breakdown."
    )

    names = [r[0] for r in shap_ranking]
    values = [r[1] for r in shap_ranking]
    colors = ["#C0862E" if n == "IMD Decile" else "#1B2A4A" for n in names]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(
        xaxis_title="Mean |SHAP value| (aggregated across categories)",
        yaxis=dict(autorange="reversed"),
        height=420, margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        f"**Accommodation Type ranks first**, ahead of Age, Ethnicity, and Disability. "
        f"**IMD Decile, the deprivation measure, ranks fifth** (highlighted in gold above), "
        f"not first as the original hypothesis expected."
    )

    with st.expander("Why might deprivation rank lower than expected?"):
        st.write(
            "The Index of Multiple Deprivation blends seven separate domains, income, employment, "
            "education, health, crime, housing, and living environment, into a single composite "
            "score. That blending may dilute a sharper, more specific effect that a narrower measure "
            "like accommodation type can capture more directly. This interpretation is supported by "
            "Singh et al. (2019) on housing disadvantage and Deas et al. (2003) on the limitations of "
            "composite deprivation measures, both discussed in the dissertation's literature review."
        )

# ============================================================
# Page: Explore a Demographic Factor
# ============================================================
elif page == "Explore a Demographic Factor":
    st.title("Explore a Demographic Factor")
    st.write(
        "Select a demographic breakdown to see its categories, how many were classified as "
        "'high demand' relative to the median for that breakdown, and its overall SHAP importance rank."
    )

    breakdown_options = sorted(engineered["BREAKDOWN"].str.replace("England; ", "", regex=False).unique())
    choice = st.selectbox("Demographic breakdown", breakdown_options)

    subset = engineered[engineered["BREAKDOWN"] == f"England; {choice}"]
    rank_position = next((i + 1 for i, r in enumerate(shap_ranking) if r[0] == choice), None)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows for this breakdown", len(subset))
    col2.metric("SHAP importance rank", f"#{rank_position} of 8")
    col3.metric("High-demand rate", f"{subset['high_demand'].mean():.0%}")

    st.subheader(f"Categories within {choice}")
    cat_summary = subset.groupby("SECONDARY_LEVEL_DESCRIPTION").agg(
        rows=("high_demand", "count"), high_demand_rate=("high_demand", "mean")
    ).reset_index().sort_values("high_demand_rate", ascending=False)
    cat_summary.columns = ["Category", "Rows", "High-demand rate"]
    st.dataframe(
        cat_summary.style.format({"High-demand rate": "{:.0%}"}),
        width='stretch', hide_index=True,
    )

# ============================================================
# Page: About This Project
# ============================================================
elif page == "About This Project":
    st.title("About This Project")
    st.write(
        "This application is the interactive deliverable for an MSc dissertation in Data Analytics "
        "with Banking and Finance, testing whether financial deprivation predicts NHS mental health "
        "service demand more strongly than other demographic factors."
    )
    st.subheader("Data source")
    st.write(
        "NHS England Mental Health Services Data Set (MHSDS), filtered to eight demographic "
        "breakdowns and the single consistent measure 'people in contact with services at the end "
        "of the reporting period', April 2016 to March 2023."
    )
    st.subheader("Tools")
    st.write("Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly.")
    st.subheader("Author")
    st.write("Rawlings Ikechukwu Gbekei, Student number C5039449.")
    st.caption(
        "This application is being evaluated by peer volunteers as part of the dissertation's "
        "user acceptance testing. Feedback is collected anonymously via a separate short form."
    )
