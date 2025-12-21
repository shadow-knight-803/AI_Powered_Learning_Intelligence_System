import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataTransformation:
    """
    Performs feature engineering and preprocessing
    """

    TARGET_COL = "completion_status"

    categorical_cols = ["course_id"]

    numerical_cols = [
        "avg_time_spent",
        "max_time_spent",
        "avg_score",
        "min_score",
        "chapters_completed",
        "early_avg_score"
    ]

    def create_student_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert chapter-level data into student-level features
        """

        # Aggregate overall performance
        student_df = (
            df.groupby(["student_id", "course_id"])
            .agg(
                avg_time_spent=("time_spent", "mean"),
                max_time_spent=("time_spent", "max"),
                avg_score=("score", "mean"),
                min_score=("score", "min"),
                chapters_completed=("chapter_order", "count"),
                completion_status=("completion_status", "max")
            )
            .reset_index()
        )

        # Early performance (first 3 chapters)
        early_df = (
            df[df["chapter_order"] <= 3]
            .groupby(["student_id", "course_id"])
            .agg(early_avg_score=("score", "mean"))
            .reset_index()
        )

        # Merge early performance
        student_df = student_df.merge(
            early_df,
            on=["student_id", "course_id"],
            how="left"
        )

        return student_df

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline
        """

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, self.numerical_cols),
                ("cat_pipeline", cat_pipeline, self.categorical_cols)
            ]
        )

        return preprocessor

    def split_data(self, student_df: pd.DataFrame):
        """
        Split dataset into train and test sets
        """

        X = student_df.drop(columns=["student_id", self.TARGET_COL])
        y = student_df[self.TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test
