import sys
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logger
from src.exception import CustomException

app = FastAPI(
    title="Learning Intelligence AI Tool",
    description="Predict course completion and early dropout risk",
    version="1.0"
)

MODEL_PATH = "./artifacts/model/completion_model.pkl"
RISK_THRESHOLD = 0.4


@app.get("/")
def home():
    return {"message": "Learning Intelligence AI Tool is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload learner CSV data and get risk predictions
    """
    try:
        logger.info("API prediction request received")

        df = pd.read_csv(file.file)

        predictor = PredictionPipeline(
            model_path=MODEL_PATH,
            risk_threshold=RISK_THRESHOLD
        )

        predictions = predictor.predict(df)

        logger.info("API prediction successful")

        return JSONResponse(
            content={
                "status": "success",
                "records": len(predictions),
                "predictions": predictions.to_dict(orient="records")
            }
        )

    except Exception as e:
        logger.error("Error during API prediction")
        raise CustomException(e, sys)
