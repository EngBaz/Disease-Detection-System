import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix

def load_prepare_and_split(uploaded_file, test_size):
    """Load and preprocess the dataset, check for outliers, and split into train/test sets."""
    
    data = pd.read_csv(uploaded_file)

    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    missing_values = data.isnull().sum()
    st.write("Missing values per column:", missing_values)
    
    if missing_values.sum() > 0:
        data = data.dropna()
        st.write("Null values detected and removed.")
    else:
        st.write("There are no null values.")
        
    outlier_columns = []
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        if not outliers.empty:
            outlier_columns.append(col)
    
    if outlier_columns:
        st.write(f"Outliers detected in the following columns: {', '.join(outlier_columns)}")
    else:
        st.write("No significant outliers detected.")
    
    if outlier_columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data[outlier_columns])
        plt.xticks(rotation=90)
        plt.title("Boxplot of Columns with Outliers")
        st.pyplot(plt)
    
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train multiple models using pipeline and return best models with parameters."""
    
    y_train = LabelEncoder().fit_transform(y_train)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=5000),
                                {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], 'classifier__penalty': ['l1', 'l2'], 'classifier__solver': ['liblinear', 'saga']}),
        'Support Vector Machine': (SVC(),
                                   {'classifier__C': [0.1, 1, 10, 100], 'classifier__kernel': ['linear', 'rbf']}),
        'K-Nearest Neighbors': (KNeighborsClassifier(),
                                 {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']}),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1, 0.2]})
    }
    
    best_models = {}
    
    for model_name, (model, param_grid) in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
            ('classifier', model)
        ])
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = (grid_search.best_params_, grid_search.best_estimator_)
    
    return best_models

def predict_and_measure_performance(X_test, y_test, best_models):
    """Evaluate models on test data and display results in Streamlit."""
    
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    
    for model_name, (best_params, best_model) in best_models.items():
        y_pred = best_model.predict(X_test)
        f1 = f1_score(y_test_encoded, y_pred)
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        st.subheader(f"Model: {model_name}")
        st.write(f"**Best Parameters:** {best_params}")
        st.metric(label="F1-Score", value=f"{f1:.4f}")
        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

def main():
    """Streamlit UI for model training and evaluation."""
    
    test_size = 0.2
    
    st.title("Disease Prediction using ML Models")
    uploaded_file = st.file_uploader("Upload the dataset in CSV format", type=['csv'])
    
    if uploaded_file is not None:
        
        X_train, X_test, y_train, y_test = load_prepare_and_split(uploaded_file, test_size)
        best_models = train_models(X_train, y_train)
        st.write("Model training completed successfully.")
        predict_and_measure_performance(X_test, y_test, best_models)

if __name__ == "__main__":
    main()
