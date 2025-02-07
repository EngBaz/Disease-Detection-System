import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def load_prepare_and_split(uploaded_file):
    """Load and preprocess the dataset."""
    
    data = pd.read_csv(uploaded_file) 
    
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the model."""
        
    y_train = LabelEncoder().fit_transform(y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1', 'l2'],  
    'solver': ['liblinear', 'saga']
    }
    
    log_reg = LogisticRegression(random_state=42, max_iter=5000)
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    return best_params, best_model, scaler


def predict_and_measure_performance(X_test, y_test, best_params, best_model, scaler):
    """Make predictions on test data and measure performance."""
    
    X_test = scaler.transform(X_test)  
    y_pred_best = best_model.predict(X_test)

    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    accuracy_best = accuracy_score(y_test_encoded, y_pred_best)
    f1_best = f1_score(y_test_encoded, y_pred_best)
    cm_best = confusion_matrix(y_test_encoded, y_pred_best)
    report_best = classification_report(y_test_encoded, y_pred_best)

    st.write(f"Best Parameters: {best_params}")
    st.write(f"Best Logistic Regression Accuracy: {accuracy_best:.4f}")
    st.write(f"Best Logistic Regression F1-Score: {f1_best:.4f}")
    st.write(f"Best Confusion Matrix: \n{cm_best}")
    st.write(f"Best Classification Report:\n{report_best}")
    
    
def main():
    
    st.write("Upload your data to detect potential diseases")
    
    uploaded_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        X_train, X_test, y_train, y_test = load_prepare_and_split(uploaded_file)
    
        best_params, best_model, scaler = train_model(X_train, y_train)
    
        predict_and_measure_performance(X_test, y_test, best_params, best_model, scaler)
    
if __name__ == "__main__":
    main()

    
    
 

    