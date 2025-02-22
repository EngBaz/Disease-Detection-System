# Disease Prediction using Machine Learning

This project is a machine learning-based disease prediction system built with Streamlit. It preprocesses a dataset, trains multiple models, and evaluate their performance, all within an interactive web app.

## 🚀 Features

* Data Preprocessing (<code>Missing Values</code>, <code>Outlier Detection</code>, <code>Balanced/Imbalanced Dataset</code>, <code>Class Distribution Check</code>)

* Train multiple ML models using <code>GridSearchCV</code> and <code>Cross Validation</code>

* Evaluate model performance (<code>Accuracy</code>, <code>Confusion Matrix</code>, <code>Overfitting Check</code>)

* Interactive Visualization using <code>Streamlit</code>

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
- **Performance Measurement:** Accuracy and Confusion Matrix 

## 🚀 Getting Started

1. Clone this repository: <code>git clone github.com/EngBaz/Disease-Detection-System.git</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>
4. Run <code>streamlit run main.py</code>

## Results

Trained several models including <code>logistic regression</code>, <code>k-nearest neigbours</code>, <code>support vector machines</code>, and <code>XGBoost</code> using <code>GridSearchCV</code> for hyperparameter tuning.
