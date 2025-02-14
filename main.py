import streamlit as st

from prediction_utils import PredictionTechniques
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def main():
    test_size = 0.3
    cv = 6
    scoring = 'accuracy'

    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=5000),
            {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], 'classifier__penalty': ['l1', 'l2'], 'classifier__solver': ['liblinear']}
        ),
        'Support Vector Machine': (
            SVC(),
            {'classifier__C': [0.1, 1, 10, 100], 'classifier__kernel': ['linear', 'rbf']}
        ),
        'K-Nearest Neighbors': (
            KNeighborsClassifier(),
            {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']}
        ),
        'XGBoost': (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1, 0.2]}
        )
    }

    predictor = PredictionTechniques(test_size=test_size, cv=cv, scoring=scoring, models=models)

    st.title("Disease Prediction using ML Models")
    uploaded_file = st.file_uploader("Upload the dataset in CSV format", type=['csv'])

    if uploaded_file is not None:
        X_train, X_test, y_train, y_test = predictor.load_prepare_and_split(uploaded_file)
        best_models = predictor.train_models(X_train, y_train)

        if not best_models:
            st.error("⚠️ No models were successfully trained!")
        else:
            st.write("✅ Model training completed successfully.")
            predictor.predict_and_measure_performance(X_test, y_test, best_models)
            predictor.check_overfitting(X_train, y_train, X_test, y_test, best_models)

if __name__ == "__main__":
    main()
