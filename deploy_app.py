import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

st.set_page_config(page_title="Solar Power ML App", layout="wide")

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🔆 Solar Power Generation")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded = st.file_uploader("Upload Solar Power Dataset (.csv)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # ================================================================
    #                   EDA SECTION
    # ================================================================
    st.header("📊 Exploratory Data Analysis")

    # Missing values
    st.subheader("🔍 Missing Values")
    st.write(numeric_df.isna().sum())

    # Summary statistics
    st.subheader("📘 Summary Statistics")
    st.write(numeric_df.describe())

    # Plotly — Interactive Histogram
    st.subheader("📈 Plotly Interactive Histogram")
    col_to_plot = st.selectbox("Select feature for histogram:", numeric_df.columns)

    fig_hist = px.histogram(df, x=col_to_plot, nbins=30, marginal="box", title=f"Distribution of {col_to_plot}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Plotly — Scatter Plot
    st.subheader("📈 Plotly Interactive Scatter Plot")
    x_col = st.selectbox("X-axis:", numeric_df.columns)
    y_col = st.selectbox("Y-axis:", numeric_df.columns, index=1)

    fig_scatter = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Heatmap (static)
    st.subheader("📌 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # ================================================================
    #                   MACHINE LEARNING PIPELINE
    # ================================================================
    st.header("⚙️ Machine Learning — Regression Models")

    df = numeric_df.fillna(numeric_df.mean())

    target = st.selectbox("Select Target Variable", numeric_df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    test_size_val = st.slider("Test Size (%)", 10, 50, 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_val / 100, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.01),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = [r2, mae, rmse]

    results_df = pd.DataFrame(results, index=["R2", "MAE", "RMSE"]).T
    st.subheader("📊 Model Performance")
    st.dataframe(results_df)

    best_model_name = results_df["R2"].idxmax()
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    best_model = models[best_model_name]

    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(scaler, "scaler.joblib")


    # ================================================================
    #                   PREDICTION
    # ================================================================
    st.header("🔮 Prediction Using Best Model")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

    if st.button("Predict Output"):
        user_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(user_df)
        pred = best_model.predict(scaled_input)
        st.success(f"🌟 Predicted {target}: **{pred[0]:.2f}**")

    # Download model
    with open("best_model.joblib", "rb") as f:
        st.download_button("⬇ Download Best Model", f, file_name="best_model.joblib")
