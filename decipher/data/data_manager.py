import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Literal

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

logger = logging.getLogger(__name__)
_parquet_engine_types = Literal["auto", "pyarrow", "fastparquet"]
"""Engine to use for parquet IO."""


def _get_base_df(screening_path: Path, dob_path: Path) -> pd.DataFrame:
    """Read and clean raw data, and add birthday.

    Discussion:
        We extracted this from the `DataManager.read_from_csv` in order to have
        the intermediate DataFrames' memory freed as soon as possible.
    """
    raw_data = read_raw_df(screening_data_path=screening_path)
    logger.debug("Read raw data")

    base_pipeline = get_base_pipeline(
        birthday_file=dob_path, drop_missing_birthday=True
    )

    return base_pipeline.fit_transform(raw_data)


class DataManager:
    @classmethod
    def read_from_csv(cls, screening_path: Path, dob_path: Path):
        base_df = _get_base_df(screening_path, dob_path)
        logger.debug("Got base DF")
        exams = Pipeline(
            [
                ("wide_to_long", ToExam()),
                ("age_adder", AgeAdder(date_field="exam_date", birth_field="FOEDT")),
                ("risk_mapper", RiskAdder()),
            ],
            verbose=True,
        ).fit_transform(base_df)
        logger.debug("Got exams DF")
        observation_data_transformer = ObservationMatrix()
        screening_data: pd.DataFrame = observation_data_transformer.fit_transform(exams)
        pid_to_row = observation_data_transformer.pid_to_row
        logger.debug("Got observations DF")
        person_df: pd.DataFrame = PersonStats(base_df=base_df).fit_transform(exams)
        logger.debug("Got person DF")
        return DataManager(
            screening_data=screening_data, person_df=person_df, pid_to_row=pid_to_row
        )

    def __init__(
        self,
        screening_data: pd.DataFrame,
        person_df: pd.DataFrame,
        pid_to_row: dict[int, int],
    ):
        self.screening_data = screening_data
        self.person_df = person_df
        self.pid_to_row = pid_to_row

    def save_to_parquet(
        self, directory: Path, engine: _parquet_engine_types = "auto"
    ) -> None:
        if not directory.is_dir():
            raise ValueError()
        self.screening_data.to_parquet(
            directory / "screening_data.parquet", engine=engine
        )
        self.person_df.to_parquet(directory / "person_df.parquet", engine=engine)
        with open(directory / "pid_to_row.json", "w") as file:
            json.dump(self.pid_to_row, file)
        with open(directory / "metadata.json", "w") as file:
            json.dump({"decipher_version": version("decipher")}, file)

    @classmethod
    def from_parquet(
        cls,
        directory: Path,
        engine: _parquet_engine_types = "auto",
        ignore_decipher_version: bool = False,
    ) -> "DataManager":
        if not directory.is_dir():
            raise ValueError()
        with open(directory / "metadata.json") as file:
            metadata = json.load(file)
        if (file_version := metadata["decipher_version"]) != (
            current_version := version("decipher")
        ):
            message = f"The file you are reading was made with decipher version {file_version}. The current version is {current_version}."
            if not ignore_decipher_version:
                raise ValueError(
                    message + "Set ignore_decipher_version=True to ignore this"
                )
            else:
                logger.warning(message)
        screening_data = pd.read_parquet(
            directory / "screening_data.parquet", engine=engine
        )
        person_df = pd.read_parquet(directory / "person_df.parquet", engine=engine)
        with open(directory / "pid_to_row.json") as file:
            pid_to_row = json.load(file)
        # json will make the key a str, so explicitly convert to int
        pid_to_row = {int(key): value for key, value in pid_to_row.items()}
        return DataManager(
            screening_data=screening_data, person_df=person_df, pid_to_row=pid_to_row
        )

    def shape(self) -> tuple[int, int]:
        n_rows = self.screening_data["row"].max() + 1
        n_cols = self.screening_data["bin"].max() + 1
        return (n_rows, n_cols)

    def data_as_coo_array(self) -> coo_array:
        clean_screen = self.screening_data[["risk", "row", "bin"]].dropna()
        array = coo_array(
            (
                clean_screen["risk"],
                (clean_screen["row"], clean_screen["bin"]),
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
        features["row"] = features["PID"].map(self.pid_to_row)
        features["col_index"] = features["variable"].map(
            {col: i for i, col in enumerate(cols)}
        )

        return coo_array(
            (features["value"], (features["row"], features["col_index"])),
            shape=(n_rows, n_cols),
            dtype="int8",
        )
