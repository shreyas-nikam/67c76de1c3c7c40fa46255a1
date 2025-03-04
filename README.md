# Project Title: QuCreate Model Residual Explorer

## Description

The **QuCreate Model Residual Explorer** is a Streamlit application designed to enhance your understanding of machine learning model performance through interactive visualization and exploration of model residuals. This tool allows you to:

- **Select Task Type:** Choose between Regression and Classification tasks to analyze different types of models.
- **Choose Models:** Explore residuals from various models, including Linear Regression, Decision Tree Regressor for regression tasks, and Logistic Regression, Decision Tree Classifier for classification tasks.
- **Visualize Residuals:**  Generate insightful visualizations of model residuals using scatter plots and histograms.
- **Color-Code Residuals:** In scatter plots, color-code residuals by feature values to identify potential relationships between feature values and residual patterns.
- **Evaluate Model Performance:** Review standard model validation metrics such as Mean Squared Error (MSE) for regression and Accuracy, Classification Report, and Confusion Matrix for classification.
- **Explore Feature Importance:** Gain basic insights into model explainability by viewing feature importance plots (when available for the selected model).

This application utilizes a synthetic dataset for demonstration purposes, making it easy to experiment and learn about residual analysis without needing to upload or process your own data initially. It is an excellent tool for educational purposes, model debugging, and gaining a deeper intuition about model behavior.

## Installation

To run the QuCreate Model Residual Explorer, you need to have Python installed on your system.  It is recommended to use Python 3.8 or higher. Follow these steps for installation:

1.  **Install Python:** If you don't have Python installed, download the latest version of Python 3 from the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/). Follow the installation instructions for your operating system.

2.  **Verify pip Installation:** Pip is the package installer for Python. Ensure pip is installed by opening your terminal or command prompt and running:

    ```bash
    pip --version
    ```

    If pip is not installed, you may need to install it separately. Refer to the official pip documentation for installation instructions: [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/)

3.  **Install Required Python Packages:** Open your terminal or command prompt and install the necessary Python libraries using pip. Run the following command:

    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn modeva
    ```

    This command will install:
    - `streamlit`: For creating the interactive web application.
    - `pandas`: For data manipulation and analysis.
    - `numpy`: For numerical operations.
    - `matplotlib`: For plotting.
    - `seaborn`: For enhanced statistical data visualization.
    - `scikit-learn`: For machine learning metrics.
    - `modeva`: A library for model validation and evaluation (used in this application).

## Usage

1.  **Save the Python Script:** Save the provided Python code as a `.py` file (e.g., `residual_explorer.py`).

2.  **Run the Streamlit Application:** Open your terminal or command prompt, navigate to the directory where you saved the Python script, and run the following command:

    ```bash
    streamlit run residual_explorer.py
    ```

    Replace `residual_explorer.py` with the actual name of your saved file if you chose a different name.

3.  **Access the Application in Your Browser:** Streamlit will automatically launch the application in your default web browser. If it doesn't open automatically, you will see a URL in your terminal (usually `http://localhost:8501`). Open this URL in your browser to access the Model Residual Explorer.

4.  **Interact with the Application:**
    - **Sidebar Controls:** Use the sidebar on the left to customize your analysis:
        - **Select Task Type:** Choose "Regression" or "Classification" from the dropdown menu.
        - **Select Model:** Choose a specific model type relevant to your selected task (e.g., "Linear Regression" for Regression).
        - **Color code residuals by Feature (for scatter plots):** Select a feature from the dropdown to color-code the data points in the residual scatter plot based on the values of this feature. This helps in identifying potential patterns related to specific features.
        - **Visualization Type for Residuals:** Choose between "Scatter Plot" and "Histogram" to visualize the residuals in different ways. Scatter plots are useful for observing patterns against predicted values, while histograms show the distribution of residuals.

    - **Main Panel:** The main panel will dynamically update based on your selections:
        - **Synthetic Dataset Generation:** Displays a sample of the synthetic dataset being used.
        - **Model Training and Residual Analysis for [Task Type]:**  Shows the selected task type. The application then trains the chosen model, calculates residuals, and presents visualizations and metrics.
        - **Residual Visualizations:** Displays the chosen visualization type (Scatter Plot or Histogram) of the residuals.
            - **Scatter Plot:**  Ideally, residuals should be randomly scattered around the horizontal zero line. Patterns or trends in the scatter plot may indicate issues with the model. Color-coding by feature can reveal if certain feature values are associated with larger or systematic residuals.
            - **Histogram:**  A histogram shows the distribution of residuals. For a well-performing model, residuals should ideally be normally distributed around zero, indicating that the errors are random and unbiased.
        - **Model Validation Metrics:** Presents performance metrics relevant to the selected task:
            - **Regression:** Displays Mean Squared Error (MSE). Lower MSE generally indicates better model fit.
            - **Classification:** Displays Accuracy, a Classification Report (including precision, recall, F1-score for each class), and a Confusion Matrix. These metrics provide a comprehensive view of classification performance.
        - **Model Explainability (Basic - Feature Importance):** If the selected model supports feature importance interpretation via `modeva`, a plot showing feature importances will be displayed. This can offer insights into which features are most influential in the model's predictions, and indirectly, potentially related to residual patterns.

5.  **Explore and Interpret Results:** Analyze the visualizations and metrics to understand the performance of the chosen model and identify potential areas for improvement. Pay attention to patterns in residual plots and the values of the validation metrics.

## Credits

Developed by QuantUniversity.

For more information about QuantUniversity and our educational resources in quantitative finance and data science, please visit our website: [https://www.quantuniversity.com/](https://www.quantuniversity.com/)

## License

© 2025 QuantUniversity. All Rights Reserved.

This application is provided for educational and demonstration purposes only. It is not intended for commercial use without explicit permission from QuantUniversity.  All rights to this software and its content are reserved by QuantUniversity. Reproduction, redistribution, or modification of this application without prior written consent from QuantUniversity is strictly prohibited.

For inquiries regarding licensing or commercial use, please contact QuantUniversity through our website.
