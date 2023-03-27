from pathlib import Path

import numpy.typing as npt
import pandas as pd
from scipy.sparse import coo_array
from sklearn.pipeline import Pipeline

from decipher.data.util import prediction_data
from decipher.processing.pipeline import get_base_pipeline, read_raw_df
from decipher.processing.transformers import (
    AgeAdder,
    ObservationMatrix,
    PersonStats,
    RiskAdder,
    ToExam,
)


class DataManager:
    def __init__(self, screening_path: Path, dob_path: Path) -> None:
        raw_data = read_raw_df(screening_data_path=screening_path)

        self.base_pipeline = get_base_pipeline(
            birthday_file=dob_path, drop_missing_birthday=True
        )

        base_df = self.base_pipeline.fit_transform(raw_data)
        self.exams = Pipeline(
            [
                ("wide_to_long", ToExam()),
                ("age_adder", AgeAdder(date_field="exam_date", birth_field="FOEDT")),
                ("risk_mapper", RiskAdder()),
            ],
            verbose=True,
        ).fit_transform(base_df)
        self.observation_data_transformer = ObservationMatrix()
        self.screening_data: pd.DataFrame = (
            self.observation_data_transformer.fit_transform(self.exams)
        )
        self.person_df: pd.DataFrame = PersonStats(base_df=base_df).fit_transform(
            self.exams
        )

    def shape(self) -> tuple[int, int]:
        n_rows = self.screening_data["row"].max() + 1
        n_cols = self.screening_data["bin"].cat.codes.max() + 1
        return (n_rows, n_cols)

    def data_as_coo_array(self) -> coo_array:
        clean_screen = self.screening_data[["risk", "row", "bin"]].dropna()
        array = coo_array(
            (
                clean_screen["risk"],
                (clean_screen["row"], clean_screen["bin"].cat.codes),
            ),
            shape=self.shape(),
            dtype="int8",
        )
        array.eliminate_zeros()
        return array

    def get_masked_data(self) -> tuple[coo_array, npt.NDArray, npt.NDArray]:
        X = self.data_as_coo_array()
        masked_X, t_pred, y_true = prediction_data(X.toarray())
        masked_X = coo_array(masked_X)
        masked_X.eliminate_zeros()  # type: ignore[attr-defined]
        return masked_X, t_pred, y_true

    def feature_data_as_coo_array(self, cols: list[str] | None = None) -> coo_array:
        cols = cols or ["has_positive", "has_negative"]

        n_rows = self.shape()[0]
        n_cols = len(cols)

        features = (
            self.person_df.reset_index().melt(id_vars="PID", value_vars=cols).dropna()
        )
        features["row"] = features["PID"].map(
            self.observation_data_transformer.pid_to_row
        )
        features["col_index"] = features["variable"].map(
            {col: i for i, col in enumerate(cols)}
        )

        return coo_array(
            (features["value"], (features["row"], features["col_index"])),
            shape=(n_rows, n_cols),
            dtype="int8",
        )
