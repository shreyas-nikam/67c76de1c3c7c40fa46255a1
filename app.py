import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="QuCreate Model Residual Explorer Lab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("Model Residual Explorer")
st.sidebar.divider()

st.title("Model Residual Explorer - QuLab")
st.markdown("## Overview")
st.write(
    "This application is designed to help you explore model residuals for both classification and regression tasks. "
    "By visualizing residuals, you can gain insights into model performance, identify weaknesses, and assess model robustness. "
    "This lab uses a synthetic dataset to demonstrate these concepts in an interactive and educational manner."
)
st.divider()

# Task selection
task_type = st.radio("Select Task Type:", ["Regression", "Classification"])

st.markdown("## 1. Synthetic Dataset Generation")
st.write("We are generating a synthetic dataset for demonstration purposes. This dataset includes numerical and categorical features to simulate real-world scenarios.")

@st.cache_data
def generate_synthetic_data(task):
    """Generates a synthetic dataset for regression or classification."""
    np.random.seed(42)
    n_samples = 200
    if task == "Regression":
        X = pd.DataFrame({
            'feature_1': np.random.rand(n_samples) * 10,
            'feature_2': np.random.randn(n_samples),
            'feature_cat': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        X = pd.get_dummies(X, columns=['feature_cat'], drop_first=True)
        y = 2 * X['feature_1'] + 0.5 * X['feature_2'] + (X['feature_cat_B'] * 3) + (X['feature_cat_C'] * -2) + np.random.randn(n_samples) * 2
        y_pred = 2 * X['feature_1'] + 0.5 * X['feature_2'] + (X['feature_cat_B'] * 3) + (X['feature_cat_C'] * -2)

    elif task == "Classification":
        X = pd.DataFrame({
            'feature_1': np.random.rand(n_samples) * 5 - 2.5,
            'feature_2': np.random.randn(n_samples),
            'feature_cat': np.random.choice(['X', 'Y'], n_samples)
        })
        X = pd.get_dummies(X, columns=['feature_cat'], drop_first=True)
        probability = 1 / (1 + np.exp(-(1.5 * X['feature_1'] + 0.8 * X['feature_2'] - 1.0*X['feature_cat_Y'])))
        y = np.random.binomial(1, probability)
        y_pred_proba = probability
        y_pred = np.round(y_pred_proba).astype(int)
        y_pred = y_pred.flatten()
        y_pred_proba = y_pred_proba.flatten()
        y_pred = pd.Series(y_pred)
        y_pred_proba = pd.Series(y_pred_proba)
        y = pd.Series(y)


    residuals = y - y_pred
    if task == "Regression":
        return X, y, y_pred, residuals
    elif task == "Classification":
        return X, y, y_pred, residuals, y_pred_proba

if task_type == "Regression":
    X, y, y_pred, residuals = generate_synthetic_data(task_type)
    st.dataframe(X.head())
    st.write("Synthetic Regression Target Variable (y) distribution:")
    st.bar_chart(y.value_counts())

elif task_type == "Classification":
    X, y, y_pred, residuals, y_pred_proba = generate_synthetic_data(task_type)
    st.dataframe(X.head())
    st.write("Synthetic Classification Target Variable (y) distribution:")
    st.bar_chart(y.value_counts())

st.divider()

st.markdown("## 2. Model Simulation & Residual Calculation")
st.write("For educational purposes, we simulated a simple model. In a real application, this would be replaced by your pre-trained ML model.")
st.write("Residuals are calculated as the difference between the actual values (y) and the predicted values (y_pred).")

if task_type == "Regression":
    st.markdown("### Regression: Predicted vs. Actual Values")
    fig_pred_actual, ax_pred_actual = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=y, ax=ax_pred_actual)
    ax_pred_actual.set_xlabel("Predicted Values")
    ax_pred_actual.set_ylabel("Actual Values")
    ax_pred_actual.set_title("Predicted vs. Actual Values (Regression)")
    st.pyplot(fig_pred_actual)
    st.write("This scatter plot visualizes how well the predicted values align with the actual values. Ideally, points should cluster closely around a diagonal line, indicating good model fit.")

    st.markdown("### Histogram of Residuals")
    fig_hist_res, ax_hist_res = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax_hist_res)
    ax_hist_res.set_xlabel("Residuals")
    ax_hist_res.set_ylabel("Frequency")
    ax_hist_res.set_title("Histogram of Residuals (Regression)")
    st.pyplot(fig_hist_res)
    st.write("This histogram shows the distribution of residuals. For a good model, residuals should be approximately normally distributed around zero, indicating unbiased predictions.")

    feature_choice = st.selectbox("Select feature for Residual vs. Feature plot:", X.columns.tolist())
    st.markdown(f"### Residuals vs. {feature_choice}")
    fig_res_feature, ax_res_feature = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X[feature_choice], y=residuals, ax=ax_res_feature)
    ax_res_feature.set_xlabel(feature_choice)
    ax_res_feature.set_ylabel("Residuals")
    ax_res_feature.set_title(f"Residuals vs. {feature_choice} (Regression)")
    st.pyplot(fig_res_feature)
    st.write(f"This scatter plot helps identify if there are patterns in residuals related to the selected feature '{feature_choice}'. Randomly scattered residuals indicate no feature-dependent bias.")

    st.markdown("### Model Validation Metrics")
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'R-squared (R2)'],
        'Value': [mse, mae, r2]
    })
    st.dataframe(metrics_df)
    st.write("These metrics provide a quantitative assessment of the model's performance. Lower MSE and MAE and R-squared closer to 1 indicate a better model for regression tasks.")


elif task_type == "Classification":
    st.markdown("### Classification: Predicted Probabilities Histogram")
    fig_proba_hist, ax_proba_hist = plt.subplots(figsize=(8, 6))
    sns.histplot(y_pred_proba, kde=False, ax=ax_proba_hist)
    ax_proba_hist.set_xlabel("Predicted Probabilities for Class 1")
    ax_proba_hist.set_ylabel("Frequency")
    ax_proba_hist.set_title("Histogram of Predicted Probabilities (Classification)")
    st.pyplot(fig_proba_hist)
    st.write("This histogram displays the distribution of predicted probabilities for the positive class. It helps to understand the model's confidence in its predictions.")


    st.markdown("### Histogram of Residuals (Classification)")
    fig_hist_res_class, ax_hist_res_class = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax_hist_res_class)
    ax_hist_res_class.set_xlabel("Residuals (y - y_pred)")
    ax_hist_res_class.set_ylabel("Frequency")
    ax_hist_res_class.set_title("Histogram of Residuals (Classification)")
    st.pyplot(fig_hist_res_class)
    st.write("For classification, residuals are the difference between actual and predicted class labels. Ideally, residuals should be centered around zero, indicating balanced predictions.")

    feature_choice_class = st.selectbox("Select feature for Residual vs. Feature plot:", X.columns.tolist())
    st.markdown(f"### Residuals vs. {feature_choice_class}")
    fig_res_feature_class, ax_res_feature_class = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X[feature_choice_class], y=residuals, ax=ax_res_feature_class)
    ax_res_feature_class.set_xlabel(feature_choice_class)
    ax_res_feature_class.set_ylabel("Residuals")
    ax_res_feature_class.set_title(f"Residuals vs. {feature_choice_class} (Classification)")
    st.pyplot(fig_res_feature_class)
    st.write(f"Analyzing residuals against features like '{feature_choice_class}' helps reveal feature-specific biases in classification performance.")

    st.markdown("### Model Validation Metrics")
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred_proba)

    metrics_df_class = pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Score', 'AUC-ROC'],
        'Value': [accuracy, f1, auc_roc]
    })
    st.dataframe(metrics_df_class)
    st.write("These metrics evaluate the classification model. Higher Accuracy, F1-Score, and AUC-ROC values indicate better classification performance.")


st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
