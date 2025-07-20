"""
churn_model.py

Churn prediction model pipeline for E-Commerce customer data.
Includes data loading, preprocessing, modeling, evaluation, and MLflow logging.

Author: DROSSIG - AHPHAM
Year: 2025

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

import mlflow
from mlflow.models.signature import infer_signature


def load_data(filename: str, sheet_name: str = "E Comm") -> pd.DataFrame:
    """
    Load E-Commerce customer data from an Excel file.

    Args:
        filename: Path to the Excel file.
        sheet_name: Name of the sheet containing data.

    Returns:
        pd.DataFrame: Raw data.
    """
    return pd.read_excel(filename, sheet_name=sheet_name)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data:
    - Merges similar categories.
    - Imputes missing values.
    - Converts categorical columns.
    - Drops unused columns.
    - Encodes categorical features as integers.

    Args:
        df: Raw data.

    Returns:
        pd.DataFrame: Preprocessed data ready for modeling.
    """
    # Merge similar categories
    df.loc[df['PreferredLoginDevice'] == 'Phone', 'PreferredLoginDevice'] = 'Mobile Phone'
    df.loc[df['PreferedOrderCat'] == 'Mobile', 'PreferedOrderCat'] = 'Mobile Phone'
    df.loc[df['PreferredPaymentMode'] == 'COD', 'PreferredPaymentMode'] = 'Cash on Delivery'
    df.loc[df['PreferredPaymentMode'] == 'CC', 'PreferredPaymentMode'] = 'Credit Card'

    # Convert int columns to str except CustomerID
    df2 = df.copy()
    for col in df2.columns:
        if col == 'CustomerID':
            continue
        else:
            if df2[col].dtype == 'int':
                df2[col] = df[col].astype(str)

    # Impute missing values
    df['Tenure'] = df['Tenure'].fillna(method='bfill')

    s_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df['WarehouseToHome'] = s_imp.fit_transform(pd.DataFrame(df['WarehouseToHome']))
    fill_list = df['HourSpendOnApp'].dropna()
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(
        pd.Series(np.random.choice(fill_list, size=len(df['HourSpendOnApp'].index)))
    )
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(method='ffill')

    imputer = KNNImputer(n_neighbors=2)
    df['CouponUsed'] = imputer.fit_transform(df[['CouponUsed']])
    imputer_2 = KNNImputer(n_neighbors=2)
    df['OrderCount'] = imputer_2.fit_transform(df[['OrderCount']])
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(method='bfill')

    # Drop CustomerID (not used)
    if 'CustomerID' in df:
        df.drop('CustomerID', axis=1, inplace=True)

    # Label encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    return df


def show_data_summary(df: pd.DataFrame) -> None:
    """Print duplicates, nulls, and unique value counts."""
    print("Duplicated values are: ", df.duplicated().sum())
    print("Null values are: ", df.isnull().sum())
    print("Unique values are: ", df.nunique())
    missing_pct = round((df.isnull().sum() * 100 / df.shape[0]), 2)
    print("Missing value % by column:\n", missing_pct)


def plot_correlation(df: pd.DataFrame, target: str = "Churn") -> None:
    """
    Plot correlation between features and the target.

    Args:
        df: Preprocessed dataframe.
        target: The target column.
    """
    corr_matrix = df.corr()
    churn_corr_vector = corr_matrix[target].sort_values(ascending=False)
    plt.figure(figsize=(10, 10))
    sns.barplot(x=churn_corr_vector, y=churn_corr_vector.index, palette='coolwarm')
    plt.title('Relation Between Features and target')
    plt.show()


def balance_data(X: pd.DataFrame, y: pd.Series, random_state: int=42):
    """
    Balance dataset using SMOTETomek oversampling/undersampling.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        tuple: Resampled (X, y)
    """
    smt = SMOTETomek(random_state=random_state)
    x_over, y_over = smt.fit_resample(X, y)
    print("Balanced shapes:", x_over.shape, y_over.shape)
    return x_over, y_over


def scale_data(x_train, x_test):
    """
    Scale features using MinMaxScaler.

    Args:
        x_train: Training features.
        x_test: Test features.

    Returns:
        tuple: (x_train_scaled, x_test_scaled)
    """
    MN = MinMaxScaler()
    x_train_scaled = MN.fit_transform(x_train)
    x_test_scaled = MN.transform(x_test)
    return x_train_scaled, x_test_scaled


def train_xgboost_classifier(x_train, y_train):
    """
    Train an XGBoost classifier.

    Args:
        x_train: Scaled training features.
        y_train: Training labels.

    Returns:
        XGBClassifier: Trained classifier.
    """
    model = XGBClassifier()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate classifier performance.

    Args:
        model: Trained classifier.
        x_train: Scaled training features.
        y_train: Training labels.
        x_test: Scaled test features.
        y_test: Test labels.

    Returns:
        dict: Metrics
    """
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    metrics = {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "test_roc_auc": roc_auc_score(y_test, y_pred_test),
        "classification_report": classification_report(y_test, y_pred_test, digits=5)
    }
    print(f"Training accuracy: {metrics['train_acc']}")
    print(f"Test accuracy: {metrics['test_acc']}")
    print(f"ROC AUC (test): {metrics['test_roc_auc']}")
    print(f"Classification Report:\n{metrics['classification_report']}")
    return metrics


def plot_confusion_roc(model, x_test, y_test):
    """
    Plot confusion matrix and ROC curve.

    Args:
        model: Trained classifier.
        x_test: Test features.
        y_test: Test labels.
    """
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax)
    plt.title("Confusion Matrix")
    plt.show()

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax_roc)
    plt.title("ROC Curve")
    plt.show()


def mlflow_log_model_run(model, x_train_scaled, x_test_scaled, y_train, y_test, experiment_name="MLflow Project Management"):
    """
    Log a model training run to MLflow.

    Args:
        model: Trained model.
        x_train_scaled: Scaled training data.
        x_test_scaled: Scaled test data.
        y_train: Training labels.
        y_test: Test labels.
        experiment_name: MLflow experiment name.
    """
    import os

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)

    accuracy = accuracy_score(y_test, model.predict(x_test_scaled))
    roc_auc = roc_auc_score(y_test, model.predict(x_test_scaled))
    y_pred = model.predict(x_test_scaled)

    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        signature = infer_signature(x_train_scaled, model.predict(x_train_scaled))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_classifier_model",
            signature=signature,
            input_example=x_train_scaled[:5],
            registered_model_name="Churn-XGBClassifier"
        )

        # Save and log Confusion Matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, x_test_scaled, y_test, ax=ax)
        plt.title("Confusion Matrix")
        fig_path = "confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path)
        if os.path.exists(fig_path):
            os.remove(fig_path)

        # Save and log ROC Curve
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_estimator(model, x_test_scaled, y_test, ax=ax_roc)
        plt.title("ROC Curve")
        fig_roc_path = "roc_curve.png"
        plt.savefig(fig_roc_path)
        plt.close(fig_roc)
        mlflow.log_artifact(fig_roc_path)
        if os.path.exists(fig_roc_path):
            os.remove(fig_roc_path)

        # Tags
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("author", "put-your-name-or-id")
        mlflow.set_tag("purpose", "Churn prediction")


def mlflow_load_predict(run_id: str, x_test_scaled):
    """
    Load a model from MLflow and return predictions.

    Args:
        run_id: MLflow run ID.
        x_test_scaled: Scaled test features.

    Returns:
        np.ndarray: Model predictions
    """
    model_uri = f"runs:/{run_id}/xgb_classifier_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model.predict(x_test_scaled)

# Example usage (script-style, comment out if you only want as importable module)
#
# if __name__ == "__main__":
#     df = load_data('E Commerce Dataset.xlsx')
#     show_data_summary(df)
#     df = preprocess_data(df)
#     plot_correlation(df, target="Churn")
#     X = df.drop('Churn', axis=1)
#     y = df['Churn']
#     x_over, y_over = balance_data(X, y)
#     x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.30, random_state=42)
#     x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
#     model = train_xgboost_classifier(x_train_scaled, y_train)
#     evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test)
#     plot_confusion_roc(model, x_test_scaled, y_test)
#     mlflow_log_model_run(model, x_train_scaled, x_test_scaled, y_train, y_test)
