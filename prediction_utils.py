import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

class PredictionTechniques:
    
    def __init__(self, test_size=0.2, cv=5, scoring='accuracy', models=None):
        
        self.test_size = test_size
        self.cv = cv
        self.scoring = scoring
        self.models = models if models is not None else {}

    def load_prepare_and_split(self, uploaded_file):
        
        data = pd.read_csv(uploaded_file)
        
        # drop unused columns
        data.drop(columns=['id'], inplace=True)

        # Check for missing values
        missing_values = data.isnull().sum()
        st.subheader("Missing values per column:")
        st.write(missing_values)

        # Remove missing values if they exist
        if missing_values.sum() > 0:
            data = data.dropna()
            st.write("Null values detected and removed.")
        else:
            st.write("There are no null values.")

        # Check if data is balanced or imbalanced
        class_distribution = data['diagnosis'].value_counts()
        st.subheader("Class Distribution:")
        
        # Show class distribution on a bar plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_distribution.index, y=class_distribution.values, palette="viridis")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        st.pyplot(plt)

        total_samples = class_distribution.sum()
        min_class_ratio = class_distribution.min() / total_samples
                
        if min_class_ratio < 0.3:
            st.write("Dataset is imbalanced (Minority class is less than 20% of total samples).")
        else:
            st.write("Dataset is balanced.")

        # Check if there are outliers
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
            st.subheader("Box plots for detecting outliers:")
            st.write(f"Outliers detected in: {', '.join(outlier_columns)}")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=data[outlier_columns])
            plt.xticks(rotation=90)
            plt.title("Boxplot of Columns with Outliers")
            st.pyplot(plt)
        else:
            st.write("No significant outliers detected.")

        # Perform feature selection
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']

        return train_test_split(X, y, test_size=self.test_size, random_state=42, stratify=y)

    def train_models(self, X_train, y_train):
        
        y_train = LabelEncoder().fit_transform(y_train)
        
        best_models = {}

        for model_name, (model, param_grid) in self.models.items():
            
            try:
                st.write(f"🔄 Training {model_name}...")
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                
                grid_search = GridSearchCV(pipeline, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_models[model_name] = (grid_search.best_params_, grid_search.best_estimator_)
                
                st.write(f"✅ {model_name} trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {e}")

        return best_models

    def predict_and_measure_performance(self, X_test, y_test, best_models):
        
        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(y_test)

        st.subheader("📊 Model Performance Summary")

        for model_name, (best_params, best_model) in best_models.items():
            try:
                st.write(f"🔍 Evaluating {model_name}...")

                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test_encoded, y_pred)
                cm = confusion_matrix(y_test_encoded, y_pred)

                st.subheader(f"Model: {model_name}")
                st.write(f"**Best Parameters:** {best_params}")
                st.metric(label="**Accuracy:**", value=f"{accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))
                
            except Exception as e:
                st.error(f"❌ Error evaluating {model_name}: {e}")

    def check_overfitting(self, X_train, y_train, X_test, y_test, best_models):
        
        y_train_encoded = LabelEncoder().fit_transform(y_train)
        y_test_encoded = LabelEncoder().fit_transform(y_test)

        for model_name, (_, best_model) in best_models.items():
            train_acc = accuracy_score(y_train_encoded, best_model.predict(X_train))
            test_acc = accuracy_score(y_test_encoded, best_model.predict(X_test))

            st.subheader(f"Overfitting Check - {model_name}")
            st.write(f"Training Accuracy: {train_acc:.4f}")
            st.write(f"Validation Accuracy: {test_acc:.4f}")

            if train_acc - test_acc > 0.1:
                st.write("⚠️ Potential overfitting detected.")
