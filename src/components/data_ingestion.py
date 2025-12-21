import os
import pandas as pd


class DataIngestion:
    """
    Handles loading of raw dataset
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV file
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.data_path}")

        df = pd.read_csv(self.data_path)
        return df
