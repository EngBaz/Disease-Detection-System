# Disease Prediction using Machine Learning

This project is a machine learning-based disease prediction system built with Streamlit. It allows users to upload a dataset, preprocess it, train multiple models, and evaluate their performance, all within an interactive web app.

## 🚀 Features

* Upload CSV Dataset

* Data Preprocessing (Missing values, Outlier detection, Class distribution check)

* Train multiple ML models using GridSearchCV

* Evaluate model performance (Accuracy, Confusion Matrix, Overfitting check)

* Interactive Visualization using Streamlit

## 📂 Project Structure

📁 disease-prediction-ml
├── 📄 main.py  # Streamlit UI & Application Logic
├── 📄 prediction_utils.py  # Data Preprocessing & Model Training
├── 📄 requirements.txt  # Python Dependencies
├── 📄 README.md  # Documentation
## :hammer: Technologies Used

- **Python**
- **Scikit-learn**
- **Streamlit**
- **Pandas**
- **NumPy**

## 📊 Model Details

- **Algorithms:** Logistic Regression, Support Vector Machines, K-Nearest Neighbours, XGBOOST
- **Preprocessing:** StandardScaler, OneHotEncoder
- **Handling Class Imbalance:** SMOTE
- **Performance Measurement:** F1-Score, Confusion Matrix 

## 🚀 Getting Started

1. Clone this repository: <code>git clone github.com/EngBaz/Disease-Detection-System.git</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>
4. 
