# ❤️ Heart Disease Prediction using Machine Learning
## 📖 Overview

Cardiovascular diseases are one of the leading causes of death worldwide. Early prediction can significantly improve patient outcomes and assist healthcare professionals in making informed decisions.

This project builds a machine learning classification system to predict the likelihood of heart disease using patient medical data. It demonstrates a complete data science workflow including preprocessing, model building, evaluation, and comparison.

##  Objective
Predict whether a patient is at risk of heart disease
Compare multiple machine learning models
Identify the best-performing model based on evaluation metrics

## 📂 Dataset

The dataset contains key medical attributes such as:

Age, Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Maximum Heart Rate
Exercise-Induced Angina
Oldpeak (ST Depression)

## ⚙️ Tech Stack
**Python**
**Pandas, NumPy** – Data preprocessing
**Matplotlib, Seaborn** – Data visualization
**Scikit-learn** – Machine learning models & pipelines

## Models Implemented

All models were implemented using **Scikit-learn Pipelines** to ensure clean preprocessing and consistent workflow.

Logistic Regression
K-Nearest Neighbors (KNN)
Naive Bayes
Decision Tree
Support Vector Machine (SVM)


## 🔧 Model Pipeline

pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    
    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier())
    ]),
    
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ])
}


### 🏆 Best Model
The **Support Vector Machine (SVM)** achieved the highest performance:

**Accuracy**: 88.04%
**F1 Score**: 89.62%

This indicates strong classification performance and balanced precision-recall tradeoff.

## 🔍 Key Insights
Feature scaling significantly improved model performance
SVM outperformed other models in both accuracy and F1 score
Naive Bayes also showed competitive performance
Tree-based models performed slightly lower due to dataset characteristics
## Project Workflow
Data Collection → Data Cleaning → EDA → Feature Scaling → Model Training → Evaluation → Model Selection

## Future Improvements
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Model deployment using Streamlit or Flask
Feature engineering for improved accuracy
Integration with real-time healthcare applications

## 🤝 Contribution
Contributions are welcome. Feel free to fork the repository and submit a pull request.

## 📬 Contact
If you’d like to collaborate or have any questions, feel free to connect with me on LinkedIn.

## ⭐ Support
If you found this project useful, please give it a ⭐ on GitHub.
