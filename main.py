import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def load_prepare_and_split(uploaded_file):
    """Load and preprocess the dataset."""
    data = pd.read_csv(uploaded_file)
    
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train multiple models and return best models with parameters."""
    
    y_train = LabelEncoder().fit_transform(y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=5000),
                                {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}),
        'Support Vector Machine': (SVC(),
                                   {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}),
        'K-Nearest Neighbors': (KNeighborsClassifier(),
                                 {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]})
    }
    
    best_models = {}
    
    for model_name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = (grid_search.best_params_, grid_search.best_estimator_)
    
    return best_models, scaler

def predict_and_measure_performance(X_test, y_test, best_models, scaler):
    """Evaluate models on test data and display results in Streamlit."""
    
    X_test = scaler.transform(X_test)
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    
    for model_name, (best_params, best_model) in best_models.items():
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred)
        cm = confusion_matrix(y_test_encoded, y_pred)
        report = classification_report(y_test_encoded, y_pred, output_dict=True)
        
        st.subheader(f"Model: {model_name}")
        st.write(f"**Best Parameters:** {best_params}")
        st.metric(label="Accuracy", value=f"{accuracy:.4f}")
        st.metric(label="F1-Score", value=f"{f1:.4f}")
        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).transpose())

def main():
    st.title("Disease Prediction using ML Models")
    uploaded_file = st.file_uploader("Upload the dataset in CSV format", type=['csv'])
    
    if uploaded_file is not None:
        X_train, X_test, y_train, y_test = load_prepare_and_split(uploaded_file)
        best_models, scaler = train_models(X_train, y_train)
        predict_and_measure_performance(X_test, y_test, best_models, scaler)

if __name__ == "__main__":
    main()
