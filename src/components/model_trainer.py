import sys
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class ModelTrainer:
    """
    Handles model training, evaluation, and persistence
    """

    def __init__(self, preprocessor):
        try:
            self.preprocessor = preprocessor
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
            self.clf = None
            logger.info("ModelTrainer initialized successfully")
        except Exception as e:
            logger.error("Error initializing ModelTrainer")
            raise CustomException(e, sys)

    def train(self, X_train, y_train):
        """
        Train the ML pipeline
        """
        try:
            logger.info("Starting model training")

            self.clf = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", self.model)
            ])

            self.clf.fit(X_train, y_train)

            logger.info("Model training completed successfully")
            return self.clf

        except Exception as e:
            logger.error("Error occurred during model training")
            raise CustomException(e, sys)

    def evaluate(self, X, y, threshold: float = 0.5):
        """
        Evaluate trained model using probability threshold
        """
        try:
            logger.info(f"Evaluating model with threshold={threshold}")

            if self.clf is None:
                raise ValueError("Model has not been trained yet")

            y_prob = self.clf.predict_proba(X)[:, 1]
            y_pred = (y_prob > threshold).astype(int)

            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1_score": f1_score(y, y_pred)
            }

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error("Error occurred during model evaluation")
            raise CustomException(e, sys)

    def save_model(self, model_path: str):
        """
        Save trained model pipeline to disk
        """
        try:
            if self.clf is None:
                raise ValueError("No trained model available to save")

            logger.info(f"Saving model to {model_path}")
            save_object(model_path, self.clf)
            logger.info("Model saved successfully")

        except Exception as e:
            logger.error("Error occurred while saving the model")
            raise CustomException(e, sys)
