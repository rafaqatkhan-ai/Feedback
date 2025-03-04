import streamlit as st
import pandas as pd
import numpy as np
import requests
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
from scipy.stats import ttest_ind, sem, t

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

# Function to perform t-tests between models
def compare_models(results):
    models = list(results.keys())
    scores = {model: results[model]['Accuracy'] for model in models}
    comparisons = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            t_stat, p_value = ttest_ind(scores[model1], scores[model2])
            comparisons[f"{model1} vs {model2}"] = {'T-statistic': t_stat, 'P-value': p_value}
    return comparisons

st.title("📚 🎓 EduPredict 🎓 📚 Boosting academic intelligence through AI")

st.sidebar.header("Select Dataset")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded successfully!")
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
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
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
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }

    if st.button("Train Models"):
        st.write("Starting training... ⏳")
        try:
            X, y = preprocess_data(df)
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
            model_results = train_models(X_train, X_test, y_train, y_test)
            st.subheader("Evaluation Results for Machine Learning Models")
            for model, metrics in model_results.items():
                mean_acc, lower_ci, upper_ci = compute_confidence_interval([metrics["Accuracy"]])
                st.write(f"**{model}** - Accuracy: {mean_acc:.3f} (95% CI: {lower_ci:.3f}-{upper_ci:.3f})")
                st.write(metrics)
            dnn_results = train_dnn(X_train, X_test, y_train, y_test)
            mean_acc, lower_ci, upper_ci = compute_confidence_interval([dnn_results["Accuracy"]])
            st.subheader("Evaluation Results for Deep Neural Network")
            st.write(f"Accuracy: {mean_acc:.3f} (95% CI: {lower_ci:.3f}-{upper_ci:.3f})")
            st.write(dnn_results)
            comparisons = compare_models(model_results)
            st.subheader("Model Comparisons (T-Test)")
            for comp, vals in comparisons.items():
                st.write(f"{comp}: T-statistic = {vals['T-statistic']:.3f}, P-value = {vals['P-value']:.3f}")
                if vals['P-value'] < 0.05:
                    st.write("✅ Statistically significant difference!")
                else:
                    st.write("❌ No significant difference.")
            st.success("🎉 Training Completed Successfully!")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
