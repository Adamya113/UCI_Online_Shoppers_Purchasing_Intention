# Online Shoppers Purchasing Intention – Classification

## a. Problem statement
The objective is to predict whether an online shopping session results in a purchase (`Revenue`). This is a binary classification problem. The application accepts a labelled test CSV containing all original input features and the `Revenue` target. It loads five previously trained classifiers, applies the saved fitted preprocessor, calculates all six evaluation metrics for every model, and displays a confusion matrix and classification report for the selected model. It never retrains on uploaded test data.

## b. Dataset description
- **Dataset:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository, dataset ID 468
- **Instances:** 12,330 sessions
- **Input features:** 17 attributes (numerical and categorical)
- **Target:** `Revenue` (Boolean; converted to 0/1)
- **Class distribution reported by UCI:** 10,422 sessions (84.5%) are negative and 1,908 (15.5%) are positive.
- **Missing values:** The UCI metadata reports no missing values.
- **Split:** 80% training / 20% testing, stratified, `random_state=42`.
- **Preprocessing:** StandardScaler for numerical columns; OneHotEncoder with `handle_unknown="ignore"` for categorical columns. The fitted ColumnTransformer is saved and reused by the app.

Dataset source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## c. Models used

The following scores are the actual results recorded in the submitted training notebook.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8812 | 0.8876 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8938 | 0.9060 | 0.6974 | 0.5550 | 0.6181 | 0.5622 |
| KNN | 0.8731 | 0.7818 | 0.6456 | 0.4005 | 0.4943 | 0.4422 |
| Naive Bayes | 0.6736 | 0.7932 | 0.2937 | 0.7880 | 0.4279 | 0.3234 |
| Support Vector Machine | 0.8869 | 0.8738 | 0.7068 | 0.4607 | 0.5578 | 0.5112 |


### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy (0.8812) and AUC (0.8876) are strong, but recall (0.3560) is relatively low. It misses a substantial share of purchasing sessions. |
| Decision Tree | Best overall model in this experiment: highest accuracy (0.8938), AUC (0.9060), F1 (0.6181), and MCC (0.5622). It gives the strongest balance among the reported metrics. |
| KNN | Accuracy is 0.8731, but AUC (0.7818) and recall (0.4005) are lower than the leading models. It is the weakest AUC result. |
| Naive Bayes | Highest recall (0.7880), meaning it identifies the largest share of actual purchases, but precision (0.2937) and accuracy (0.6736) are low, indicating many false positives. |
| Support Vector Machine | Accuracy (0.8869) is close to the Decision Tree and it has relatively high precision (0.7068). Its recall (0.4607) and F1 (0.5578) are below the Decision Tree. |

**Overall winner for this dataset: Decision Tree**, based on the best accuracy, AUC, F1-score, and MCC in this test run. If the priority were to catch as many purchasing sessions as possible, Naive Bayes would be considered because it has the highest recall, with the trade-off of many false positives.

## Project structure

```text
online_shoppers_ml_project/
├── app.py
├── train_models.py
├── ml_assign2_sem1.ipynb
├── requirements.txt
├── README.md
├── README_report.pdf
├── test_data.csv
├── model_metrics.csv
└── model/
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── SVM.pkl
    └── preprocessor.pkl
```

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate virtual environment: `python -m venv .venv && source .venv/bin/activate` (for windows `.venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt`
5. Generate model files and held-out test data once, before launching the app: `python train_models.py`
6. Launch the app: `streamlit run app.py`

The training script downloads the dataset from UCI, uses the same split and preprocessing approach as the notebook, and creates all five saved classifiers, the fitted preprocessor, and the labelled `test_data.csv`. The Streamlit app only loads these saved artifacts and evaluates uploaded test rows; it does not train or fit anything. An internet connection is required for the one-time training/artifact-generation step.
