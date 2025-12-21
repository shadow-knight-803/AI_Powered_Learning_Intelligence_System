import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger
from src.exception import CustomException


def run_training_pipeline():
    try:
        logger.info("===== Training Pipeline Started =====")

        # --------------------------------------------------
        # 1. Data Ingestion
        # --------------------------------------------------
        logger.info("Starting data ingestion")
        ingestion = DataIngestion("./data/synthetic_learning_data.csv")
        df = ingestion.load_data()
        logger.info(f"Data ingestion completed. Shape: {df.shape}")

        # --------------------------------------------------
        # 2. Data Transformation
        # --------------------------------------------------
        logger.info("Starting data transformation")
        transformer = DataTransformation()

        student_df = transformer.create_student_features(df)
        logger.info(
            f"Student-level feature engineering completed. Shape: {student_df.shape}"
        )

        X_train, X_test, y_train, y_test = transformer.split_data(student_df)
        preprocessor = transformer.get_preprocessor()
        logger.info("Train-test split and preprocessing pipeline created")

        # --------------------------------------------------
        # 3. Model Training
        # --------------------------------------------------
        logger.info("Starting model training")
        trainer = ModelTrainer(preprocessor)
        trainer.train(X_train, y_train)
        logger.info("Model training finished")

        # --------------------------------------------------
        # 4. Model Evaluation
        # --------------------------------------------------
        logger.info("Evaluating model with risk threshold = 0.4")
        test_metrics = trainer.evaluate(X_test, y_test, threshold=0.4)
        logger.info(f"Test Metrics: {test_metrics}")

        print("\n===== Test Metrics (Threshold = 0.4) =====")
        print(test_metrics)

        # --------------------------------------------------
        # 5. Save Model
        # --------------------------------------------------
        model_path = "./artifacts/model/completion_model.pkl"
        trainer.save_model(model_path)
        logger.info(f"Model saved successfully at {model_path}")

        logger.info("===== Training Pipeline Completed Successfully =====")

    except Exception as e:
        logger.error("Error occurred in training pipeline")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
