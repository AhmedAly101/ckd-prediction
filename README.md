# Chronic Kidney Disease (CKD) Prediction

A machine learning project that predicts whether a patient has chronic kidney disease from lab results and medical history, with a simple Streamlit web app for making predictions.

## Dataset

[Chronic Kidney Disease dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) (UCI Machine Learning Repository): 400 patient records, 24 features, and a binary target (`ckd` / `notckd`). It contains both numeric features (age, blood pressure, hemoglobin, serum creatinine, ...) and categorical features (hypertension, diabetes, appetite, ...), with many missing values.

## Project Files

| File | Description |
|------|-------------|
| `Final_Project_2_.ipynb` | Notebook: data cleaning, EDA, model training, tuning, and evaluation |
| `chronic_kidney_disease.arff` | The dataset |
| `app.py` | Streamlit app for entering patient data and getting a prediction |
| `ckd_model.pkl` | Trained Logistic Regression model (created by the notebook) |
| `scaler.pkl` | Fitted StandardScaler (created by the notebook) |
| `requirements.txt` | Python libraries needed to run the project |

## Workflow

1. **Loading:** parse the `.arff` file and replace `?` with missing values.
2. **EDA:** missing values, class balance, box plots, and a correlation heatmap.
3. **Preprocessing:** stratified 80/20 train/test split, median imputation for numeric columns, mode imputation for categorical columns, label encoding, and standard scaling (fitted on the training set only).
4. **Modeling:** six models were compared with 5-fold cross-validation: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and Gradient Boosting.
5. **Tuning:** `GridSearchCV` on Logistic Regression, SVM, Random Forest, and Gradient Boosting.
6. **Deployment:** the best model (Logistic Regression) and the scaler are saved with `joblib` and used by the Streamlit app.

## Results (test set, 80 samples)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Logistic Regression | 100% | 100% | 100% | 100% |
| SVM | 98.75% | 100% | 98% | 98.99% |
| Random Forest | 97.5% | 96.15% | 100% | 98.04% |
| Gradient Boosting | 97.5% | 96.15% | 100% | 98.04% |

> The dataset is small (400 records from a single hospital), so these scores may not hold on new data from other populations.

## Installation

```bash
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME
pip install -r requirements.txt
```

## Run the App

Make sure `ckd_model.pkl` and `scaler.pkl` are in the same folder as `app.py`, then run:

```bash
streamlit run app.py
```

The app will open in your browser. Enter the patient's data, press **Predict**, and you'll see the result and the probability of CKD.

## Re-training the Model

Open `Final_Project_2_.ipynb` (it was written for Google Colab), run all cells up to the one that saves the model, and it will regenerate `ckd_model.pkl` and `scaler.pkl`.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit

## Disclaimer

This project is for educational purposes only. It is not a diagnostic tool and does not replace a qualified doctor.

## Author

[Ahmed Aly]
