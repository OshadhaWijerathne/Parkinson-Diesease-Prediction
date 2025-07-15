#  Parkinson's Disease Detection Model

This project aims to build a machine learning model to predict the presence of Parkinson's disease using a dataset of biomedical voice measurements. The process involves comprehensive data preprocessing, feature selection, and model training to achieve a reliable classification model.

---

##  Project Workflow

The project is divided into two main parts, each contained within its own Jupyter Notebook:

1.  **`eda.ipynb` - Exploratory Data Analysis & Preprocessing:**
    This notebook handles all the steps required to clean and prepare the data for modeling.
    * **Data Loading:** The `parkinson_disease.csv` dataset is loaded.
    * **Data Aggregation:** Multiple voice recordings for each individual are averaged to create a single, representative data point per person.
    * **Multicollinearity Removal:** Features with a high correlation ( > 0.7) are removed to reduce redundancy.
    * **Feature Scaling:** The data is normalized using `MinMaxScaler` to scale all features between 0 and 1.
    * **Feature Selection:** `SelectKBest` with the `mutual_info_classif` score is used to identify the **top 30 most influential features**.
    * **Class Imbalance Handling:** The training data suffers from class imbalance. This is addressed using `RandomOverSampler` to create a balanced training set.
    * **Data Saving:** The final processed datasets (`X_train_resampled.csv`, `X_val.csv`, etc.) are saved for the model training phase.

2.  **`model_training.ipynb` - Model Training & Evaluation:**
    This notebook focuses on training several models and selecting the best one.
    * **Model Selection:** Three different classifiers are trained and evaluated:
        * Logistic Regression
        * XGBoost Classifier
        * Support Vector Machine (SVC)
    * **Evaluation:** The models are evaluated based on their performance on the validation set, primarily using the ROC AUC score.
    * **Best Model:** The **XGBoost Classifier** was identified as the best-performing model.
    * **Model Saving:** The trained XGBoost model is saved to a file named `trained_xgb_model.joblib` using `joblib` for future use.

---

##  Results

The XGBoost model demonstrated the best performance on the unseen validation data.

* **Validation ROC-AUC Score:** **0.76**
* **Validation Accuracy:** **84%**

The classification report for the XGBoost model highlights its effectiveness, particularly in identifying patients with Parkinson's (class 1.0).
