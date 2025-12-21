import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_object(file_path: str, obj):
    """
    Save Python object to disk using joblib
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str):
    """
    Load Python object from disk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    return joblib.load(file_path)


def evaluate_classification_model(y_true, y_pred):
    """
    Return common classification metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
