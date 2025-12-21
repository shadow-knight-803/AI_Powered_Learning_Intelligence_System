import sys
import os
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.utils import load_object
from src.logger import logger
from src.exception import CustomException


class PredictionPipeline:
    """
    Loads trained model and performs inference + risk detection
    """

    def __init__(self, model_path: str, risk_threshold: float = 0.4):
        try:
            logger.info("Initializing PredictionPipeline")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            self.model = load_object(model_path)
            self.risk_threshold = risk_threshold
            self.transformer = DataTransformation()

            logger.info(
                f"PredictionPipeline initialized with threshold={risk_threshold}"
            )

        except Exception as e:
            logger.error("Error initializing PredictionPipeline")
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate completion probabilities and risk flags
        """
        try:
            logger.info("Starting prediction pipeline")

            # Feature engineering (same as training)
            student_df = self.transformer.create_student_features(df)
            logger.info(
                f"Student-level features created. Shape: {student_df.shape}"
            )

            # Prepare input features
            X = student_df.drop(
                columns=["student_id", self.transformer.TARGET_COL]
            )

            # Predict probabilities
            probabilities = self.model.predict_proba(X)[:, 1]

            # Add predictions
            student_df["completion_probability"] = probabilities
            student_df["risk_flag"] = (
                probabilities < self.risk_threshold
            ).astype(int)

            # Human-readable label
            student_df["risk_label"] = student_df["risk_flag"].map(
                {1: "High Risk", 0: "Low Risk"}
            )

            logger.info("Prediction completed successfully")

            return student_df[
                [
                    "student_id",
                    "course_id",
                    "completion_probability",
                    "risk_label"
                ]
            ]

        except Exception as e:
            logger.error("Error occurred during prediction")
            raise CustomException(e, sys)


# --------------------------------------------------
# Run prediction as a script
# --------------------------------------------------
if __name__ == "__main__":
    try:
        DATA_PATH = "./data/synthetic_learning_data.csv"
        MODEL_PATH = "./artifacts/model/completion_model.pkl"

        logger.info("Running prediction pipeline as standalone script")

        df = pd.read_csv(DATA_PATH)

        predictor = PredictionPipeline(
            model_path=MODEL_PATH,
            risk_threshold=0.4
        )

        predictions = predictor.predict(df)

        print("\n===== Risk Prediction Output =====")
        print(predictions.head())

        # Save predictions
        os.makedirs("./artifacts/reports", exist_ok=True)
        output_path = "./artifacts/reports/risk_predictions.csv"
        predictions.to_csv(output_path, index=False)

        logger.info(f"Predictions saved at {output_path}")
        print(f"\n✅ Predictions saved at {output_path}")

    except Exception as e:
        logger.error("Fatal error in prediction pipeline")
        raise CustomException(e, sys)
