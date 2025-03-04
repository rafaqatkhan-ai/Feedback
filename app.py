import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# GitHub repository details
GITHUB_USER = "rafaqatkhan-ai"
GITHUB_REPO = "learning-feedback"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/"

# Streamlit App Title
st.title("\U0001F4DA \U0001F393 EduPredict \U0001F393 \U0001F4DA\nBoosting academic intelligence through AI")

# Function to fetch CSV files from GitHub repository
def fetch_github_csv_files():
    repo_api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents"
    try:
        response = requests.get(repo_api_url)
        if response.status_code == 200:
            files = response.json()
            csv_files = [file['download_url'] for file in files if file['name'].endswith('.csv')]
            return csv_files
        else:
            st.error(f"Failed to fetch files from GitHub: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching files: {e}")
        return []

# Load dataset (either from GitHub or uploaded file)
st.sidebar.header("Select Dataset")

github_csv_files = fetch_github_csv_files()
selected_github_csv = st.sidebar.selectbox("Choose a dataset from GitHub:", ["None"] + github_csv_files)
uploaded_file = st.sidebar.file_uploader("Or Upload a CSV file", type=["csv"])

if "df" not in st.session_state:
    st.session_state.df = None

if selected_github_csv != "None":
    st.session_state.df = pd.read_csv(selected_github_csv)
    st.sidebar.success(f"Loaded dataset from GitHub: {selected_github_csv}")

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded successfully!")

if st.session_state.df is not None:
    df = st.session_state.df
    st.write("### Dataset Preview")
    st.write(df.head())

    def preprocess_data(df):
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()

        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(exclude=['object']).columns

        if len(num_cols) > 0:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])

        if len(cat_cols) > 0:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
            X_cat.columns = encoder.get_feature_names_out(cat_cols)
            X = X.drop(columns=cat_cols).reset_index(drop=True)
            X = pd.concat([X, X_cat], axis=1)

        return X, y

    def train_models(X_train, X_test, y_train, y_test):
        classifiers = {
            'AdaBoost': AdaBoostClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(),
            'LightGBM': lgb.LGBMClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0)
        }
        results = {}

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            ci = stats.norm.interval(0.95, loc=acc, scale=np.sqrt((acc * (1 - acc)) / len(y_test)))

            results[name] = {
                "Accuracy": acc,
                "Confidence Interval (95%)": ci,
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted')
            }
        return results

    def train_dnn(X_train, X_test, y_train, y_test):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
        y_pred = np.argmax(model.predict(X_test), axis=1)

        acc = accuracy_score(y_test, y_pred)
        ci = stats.norm.interval(0.95, loc=acc, scale=np.sqrt((acc * (1 - acc)) / len(y_test)))
        return {
            "Accuracy": acc,
            "Confidence Interval (95%)": ci,
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }

    if st.button("Train Models"):
        st.write("Starting training... ⏳")
        try:
            if df is None or df.empty:
                st.error("No dataset is loaded. Please upload or select a dataset!")
            else:
                X, y = preprocess_data(df)
                smote = SMOTE()
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
                model_results = train_models(X_train, X_test, y_train, y_test)
                st.subheader("Evaluation Results for Machine Learning Models")
                for model, metrics in model_results.items():
                    st.write(f"**{model}**")
                    st.write(metrics)
                dnn_results = train_dnn(X_train, X_test, y_train, y_test)
                st.subheader("Evaluation Results for Deep Neural Network")
                st.write(dnn_results)
                st.success("🎉 Training Completed Successfully!")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
