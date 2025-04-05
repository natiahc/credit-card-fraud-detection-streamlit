* `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: Contains the customer data used for training and prediction.
* `src/models/best_churn_model.pkl`: The trained machine learning model saved using pickle.
* `app.py`: The main Streamlit application code.
* `requirements.txt`: Lists the Python dependencies.
* `README.md`: This file (provides project information).
* `.gitignore`: Specifies files that Git should ignore.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your_github_repository_url>
    cd <your_repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

## Usage

1.  Navigate to the sidebar on the left.
2.  Enter the customer details using the provided input fields.
3.  Click the "Predict Churn" button.
4.  The prediction (Likely to Churn or Not Likely to Churn) and the probability of churn will be displayed in the main area.

## Model

The prediction is powered by a Gradient Boosting Classifier model trained on the provided customer churn dataset. The preprocessing steps include scaling numerical features and one-hot encoding categorical features.
