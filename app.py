import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# --- Load Data ---
DATA_URL = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
        return df
    except FileNotFoundError:
        st.error(f"Error: Could not load data from '{DATA_URL}'. Please ensure the file is in the 'data' directory.")
        return None

df = load_data()

if df is not None:
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include='object').columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # --- Preprocessing Pipeline ---
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # --- Load or Train Model ---
    MODEL_FILENAME = 'src/models/best_churn_model.pkl'

    try:
        with open(MODEL_FILENAME, 'rb') as file:
            best_model = pickle.load(file)
        st.info(f"Loaded pre-trained model from '{MODEL_FILENAME}'")
    except FileNotFoundError:
        st.warning(f"Model file '{MODEL_FILENAME}' not found. Training a Gradient Boosting model...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', GradientBoostingClassifier(random_state=42))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        # Create the 'src/models' directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(MODEL_FILENAME), exist_ok=True)
        with open(MODEL_FILENAME, 'wb') as file:
            pickle.dump(best_model, file)
        st.success(f"Trained and saved Gradient Boosting model as '{MODEL_FILENAME}'")

    # --- Streamlit App ---
    st.title("Interactive Customer Churn Prediction")
    st.markdown("Enter customer details to predict if they are likely to churn.")

    st.sidebar.header("Customer Information")
    input_data = {}
    all_features = numerical_features.tolist() + categorical_features.tolist()
    for col in all_features:
        if col in numerical_features:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            input_data[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)
        elif col in categorical_features:
            unique_values = sorted(df[col].unique().tolist())
            input_data[col] = st.sidebar.selectbox(col, unique_values)

    if st.sidebar.button("Predict Churn"):
        if best_model:
            try:
                input_df = pd.DataFrame([input_data])
                prediction_proba = best_model.predict_proba(input_df)[:, 1][0]
                prediction = "Likely to Churn" if prediction_proba > 0.5 else "Not Likely to Churn"
                confidence = f"{prediction_proba * 100:.2f}%"

                st.subheader("Prediction Result")
                if prediction == "Likely to Churn":
                    st.error(prediction)
                else:
                    st.success(prediction)
                st.write(f"Probability of Churn (Confidence): {confidence}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Model not loaded. Please ensure the model file exists or the training process completes.")

    st.markdown("---")
    st.info("This is an interactive demo for a customer churn prediction model.")
