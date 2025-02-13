import streamlit as st
from prediction_utils import PredictionTechniques


def main():
    
    test_size = 0.2
    cv = 5
    scoring = 'accuracy'
    
    predictor = PredictionTechniques(test_size=test_size, cv=cv, scoring=scoring)
    
    st.title("Disease Prediction using ML Models")
    uploaded_file = st.file_uploader("Upload the dataset in CSV format", type=['csv'])
    
    if uploaded_file is not None:
        X_train, X_test, y_train, y_test = predictor.load_prepare_and_split(uploaded_file)
        best_models = predictor.train_models(X_train, y_train)
        st.write("Model training completed successfully.")
        predictor.predict_and_measure_performance(X_test, y_test, best_models)
        predictor.check_overfitting(X_train, y_train, X_test, y_test, best_models)


if __name__ == "__main__":
    main()
