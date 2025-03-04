import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modeva import DataSet
from modeva.models import MoLinearRegression, MoLogisticRegression, MoDecisionTreeRegressor, MoDecisionTreeClassifier
from modeva import TestSuite
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import seaborn as sns

st.set_page_config(page_title="QuCreate Model Residual Explorer", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("Model Residual Explorer")
st.sidebar.markdown("Explore model residuals interactively.")
st.sidebar.divider()

st.title("Model Residual Explorer")
st.markdown("This application helps you understand model performance by visualizing and exploring model residuals. "
            "Select a task type and model to begin.")
st.divider()

# --- Data Generation ---
st.header("Synthetic Dataset Generation")
st.write("We are using a synthetic dataset for demonstration purposes. You can explore model residuals using this generated data.")

np.random.seed(42)
n_samples = 1000
features = {
    'numerical_feature_1': np.random.rand(n_samples),
    'numerical_feature_2': np.random.randn(n_samples),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
    'target_regression': 2 * np.random.rand(n_samples) + 0.5 * np.random.randn(n_samples),
    'target_classification': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
}
synthetic_df = pd.DataFrame(features)

st.dataframe(synthetic_df.head())
st.caption("First few rows of the synthetic dataset.")

# --- User Input in Sidebar ---
task_type = st.sidebar.selectbox("Select Task Type", ["Regression", "Classification"])

if task_type == "Regression":
    target_column = "target_regression"
    model_name = st.sidebar.selectbox("Select Regression Model", ["Linear Regression", "Decision Tree Regressor"])
else:
    target_column = "target_classification"
    model_name = st.sidebar.selectbox("Select Classification Model", ["Logistic Regression", "Decision Tree Classifier"])

feature_to_color_code = st.sidebar.selectbox("Color code residuals by Feature (for scatter plots)",
                                            synthetic_df.columns.tolist())

st.sidebar.divider()
st.sidebar.markdown("Adjust Visualization Parameters:")
chart_type = st.sidebar.selectbox("Visualization Type for Residuals", ["Scatter Plot", "Histogram"])

st.divider()

# --- Model Training and Prediction ---
st.header(f"Model Training and Residual Analysis for {task_type}")

ds = DataSet(name="SyntheticData")
ds.load_dataframe(synthetic_df)
ds.set_target(target_column)
ds.preprocess()
ds.set_random_split()

st.subheader("Model Training in Progress...")
model = None
if task_type == "Regression":
    if model_name == "Linear Regression":
        model = MoLinearRegression(name="demo_model")
    elif model_name == "Decision Tree Regressor":
        model = MoDecisionTreeRegressor(name="demo_tree_regressor", max_depth=3) # Limited depth for explainability
else: # Classification
    if model_name == "Logistic Regression":
        model = MoLogisticRegression(name="demo_model")
    elif model_name == "Decision Tree Classifier":
        model = MoDecisionTreeClassifier(name="demo_tree_classifier", max_depth=3) # Limited depth for explainability

if model:
    model.fit(ds.train_x, ds.train_y.ravel()) # ravel for sklearn compatibility
    st.success(f"Model ({model_name}) training completed!")

    st.subheader("Model Predictions and Residual Calculation")
    predictions = model.predict(ds.test_x)
    actual_values = ds.test_y.ravel()

    if task_type == "Regression":
        residuals = actual_values - predictions
        st.write("Residuals are calculated as: Actual Value - Predicted Value")
        st.latex(r'\text{Residual} = \text{Actual Value} - \text{Predicted Value}')
    else: # Classification
        residuals = np.array([1 if p != a else 0 for p, a in zip(predictions, actual_values)]) # 1 for misclassification, 0 for correct
        st.write("Residuals are binary for classification: 1 for misclassification, 0 for correct classification.")
        st.markdown("Residual = 1 if Prediction != Actual Value, else 0")


    st.subheader("Residual Visualizations")

    if chart_type == "Scatter Plot":
        st.write(f"Visualizing residuals in a scatter plot, color-coded by '{feature_to_color_code}'.")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(predictions, residuals, c=synthetic_df.loc[ds.test_idx, feature_to_color_code], cmap='viridis', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residual Scatter Plot Color-coded by {feature_to_color_code}")
        plt.colorbar(scatter, ax=ax, label=feature_to_color_code)
        st.pyplot(fig)
        st.caption("This scatter plot shows residuals against predicted values. Ideally, residuals should be randomly scattered around zero. "
                     "Color coding by feature helps identify if residual patterns are influenced by feature values.")

    elif chart_type == "Histogram":
        st.write("Displaying a histogram of the residuals to check their distribution.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Residuals")
        st.pyplot(fig)
        st.caption("A histogram of residuals helps visualize their distribution. For a well-performing model, residuals should ideally be normally distributed around zero.")

    st.subheader("Model Validation Metrics")
    st.write("Evaluating model performance using standard metrics.")
    if task_type == "Regression":
        mse = mean_squared_error(actual_values, predictions)
        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        st.caption("MSE measures the average squared difference between the estimated values and the actual value. Lower MSE values indicate better fit.")

    else: # Classification
        accuracy = accuracy_score(actual_values, np.round(predictions)) # Assuming binary classification and predictions are probabilities
        st.metric("Accuracy", f"{accuracy:.4%}")
        st.caption("Accuracy is the proportion of correct predictions out of the total predictions. Higher accuracy values indicate better performance.")
        report = classification_report(actual_values, np.round(predictions), output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        st.caption("Classification Report provides precision, recall, f1-score and support for each class, and overall metrics.")

        st.subheader("Confusion Matrix")
        st.write("Confusion Matrix to visualize classification performance per class.")
        cm = confusion_matrix(actual_values, np.round(predictions))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.caption("Confusion Matrix shows the counts of True Positives, True Negatives, False Positives, and False Negatives, helping understand model's class-wise performance.")


    st.subheader("Model Explainability (Basic - Feature Importance)")
    st.write("Displaying basic feature importance from the trained model (if available).")
    if hasattr(model, 'interpret_fi'):
        fi_result = model.interpret_fi(ds.train_x, ds.train_y.ravel())
        fig_fi = fi_result.plot()
        st.pyplot(fig_fi)
        st.caption("Feature Importance plot indicates which features had the most influence on the model's predictions. "
                     "While this is model-centric, it can indirectly help understand which features might be related to residuals.")
    else:
        st.warning("Feature importance plot is not available for this model type in modeva directly. ")
        st.info("Note: Feature importance is model-specific and may not be available for all model types directly through modeva's interpret_fi method.")

else:
    st.error("Model training failed. Please check model selection and dataset.")


st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
