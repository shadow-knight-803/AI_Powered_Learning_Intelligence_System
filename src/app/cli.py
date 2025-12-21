import sys
import pandas as pd
import typer

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logger
from src.exception import CustomException

app = typer.Typer(help="Learning Intelligence AI Tool (CLI)")


@app.command()
def predict(
    data_path: str = typer.Option(
        "./data/synthetic_learning_data.csv",
        help="Path to input CSV file"
    ),
    model_path: str = typer.Option(
        "./artifacts/model/completion_model.pkl",
        help="Path to trained model"
    ),
    threshold: float = typer.Option(
        0.4,
        help="Risk threshold for classification"
    ),
    output_path: str = typer.Option(
        "./artifacts/reports/risk_predictions.csv",
        help="Path to save prediction results"
    )
):
    """
    Run risk prediction from command line
    """
    try:
        logger.info("CLI prediction started")

        df = pd.read_csv(data_path)

        predictor = PredictionPipeline(
            model_path=model_path,
            risk_threshold=threshold
        )

        predictions = predictor.predict(df)

        predictions.to_csv(output_path, index=False)

        logger.info(f"CLI predictions saved at {output_path}")

        print("\n✅ Prediction completed successfully")
        print(predictions.head())

    except Exception as e:
        logger.error("Error during CLI prediction")
        raise CustomException(e, sys)


if __name__ == "__main__":
    app()
