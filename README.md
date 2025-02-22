# Disease Prediction using Machine Learning

This project is a machine learning-based disease prediction system built with Streamlit. It preprocesses a dataset, trains multiple models, and evaluate their performance, all within an interactive web app.

## 🚀 Features

* Data Preprocessing (<code>Missing values</code>, <code>outlier detection</code>, <code>balanced/imbalanced dataset</code>, <scode>class distribution check</code>)

* Train multiple ML models using </code>GridSearchCV</code> and <code>Cross Validation</code>

* Evaluate model performance (Accuracy, Confusion Matrix, Overfitting check)

* Interactive Visualization using Streamlit

## 📂 Project Structure

├── 📄 main.py  

├── 📄 prediction_utils.py 

├── 📄 requirements.txt  

├── 📄 README.md  

## :hammer: Technologies Used

- **Python**
- **Scikit-learn**
- **Streamlit**
- **Pandas**
- **NumPy**

## 📊 Model Details

- **Algorithms:** Logistic Regression, Support Vector Machines, K-Nearest Neighbours, XGBOOST
- **Preprocessing:** StandardScaler
- **Performance Measurement:** Accuracy, Confusion Matrix 

## 🚀 Getting Started

1. Clone this repository: <code>git clone github.com/EngBaz/Disease-Detection-System.git</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>
4. Run <code>streamlit run main.py</code>
