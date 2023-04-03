import itertools
import json
import logging
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

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


class PersonFilter(ABC):
    """Strategy for filtering people; given a person_df, return PIDs"""

    @abstractmethod
    def filter(self, person_df: pd.DataFrame) -> Iterable[int] | "pd.Series[int]":
        ...

    def metadata(self) -> dict:
        return {"type": "generic_filter"}


class AtLeastNNonHPV(PersonFilter):
    def __init__(self, min_n: int = 2):
        self.min_n = min_n

    def filter(self, person_df: pd.DataFrame) -> Iterable[int] | "pd.Series[int]":
        has_sufficient_screenings = (
            person_df[["histology_count", "cytology_count"]].agg("sum", axis=1)
            >= self.min_n
        )
        return has_sufficient_screenings[has_sufficient_screenings].index

    def metadata(self) -> dict:
        return {"type": "min_n_non_hpv", "min_n": self.min_n}


class TrueFields(PersonFilter):
    """Filter people where all fields in `fields` are True."""

    def __init__(self, fields: list[str]):
        self.fields = fields

    def filter(self, person_df: pd.DataFrame) -> Iterable[int] | "pd.Series[int]":
        matches = person_df[self.fields].all(axis=1)
        return person_df[matches].index

    def metadata(self) -> dict:
        return {"type": "true fields", "fields": self.fields}


class OperatorFilter(PersonFilter):
    """Arbitrary operator comparison.

    Example:
    >>> from operator import lt  # Less than
    >>> OperatorFilter("age_min", lt, timedelta(days=365 * 50)).filter(person_df)

    TODO:
        Can potentially make field a list of fields, and then apply all(axis=1),
        similar to TrueFields. Then, TrueFields would just be a special case of this.
    """

    def __init__(self, field: str, operator: Callable, value: Any):
        self.field = field
        self.operator = operator
        self.value = value

    def filter(self, person_df: pd.DataFrame) -> Iterable[int] | "pd.Series[int]":
        matches: "pd.Series[bool]" = self.operator(person_df[self.field], self.value)
        return person_df[matches].index

    def metadata(self) -> dict:
        return {
            "type": "operator filter",
            "field": self.field,
            "operator": str(self.operator),
            "value": str(self.value),
        }


class CombinePersonFilter(PersonFilter):
    def __init__(self, filters: list[PersonFilter]):
        self.filters = filters

    def filter(self, person_df: pd.DataFrame) -> Iterable[int] | "pd.Series[int]":
        return set(
            itertools.chain.from_iterable(
                (filter_.filter(person_df) for filter_ in self.filters)
            )
        )

    def metadata(self) -> dict:
        return {
            "type": "combine filter",
            "sub filters": [filter_.metadata() for filter_ in self.filters],
        }


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
        person_df: pd.DataFrame = PersonStats(base_df=base_df).fit_transform(exams)
        logger.debug("Got person DF")
        return DataManager(
            person_df=person_df,
            exams_df=exams,
        )

    def __init__(
        self,
        person_df: pd.DataFrame,
        exams_df: pd.DataFrame,
        pid_to_row: dict[int, int] | None = None,
        screening_data: pd.DataFrame | None = None,
        metadata: dict | None = None,
    ):
        self.screening_data = screening_data
        self.person_df = person_df
        self.exams_df = exams_df
        self.pid_to_row = pid_to_row
        self.metadata = metadata or {}

    def save_to_parquet(
        self, directory: Path, engine: _parquet_engine_types = "auto"
    ) -> None:
        if not directory.is_dir():
            raise ValueError()
        if self.screening_data is not None:
            self.screening_data.to_parquet(
                directory / "screening_data.parquet", engine=engine
            )
            with open(directory / "pid_to_row.json", "w") as file:
                json.dump(self.pid_to_row, file)
        self.person_df.to_parquet(directory / "person_df.parquet", engine=engine)
        self.exams_df.to_parquet(directory / "exams_df.parquet", engine=engine)
        with open(directory / "metadata.json", "w") as file:
            json.dump({"decipher_version": version("decipher")} | self.metadata, file)

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
        if (screening_file := directory / "screening_data.parquet").exists():
            screening_data = pd.read_parquet(screening_file, engine=engine)
            with open(directory / "pid_to_row.json") as file:
                pid_to_row = json.load(file)
            # json will make the key a str, so explicitly convert to int
            pid_to_row = {int(key): value for key, value in pid_to_row.items()}
        else:
            screening_data = None
            pid_to_row = None
        person_df = pd.read_parquet(directory / "person_df.parquet", engine=engine)
        exams_df = pd.read_parquet(directory / "exams_df.parquet", engine=engine)
        return DataManager(
            person_df=person_df,
            exams_df=exams_df,
            screening_data=screening_data,
            pid_to_row=pid_to_row,
        )

    def get_screening_data(
        self,
        filter_strategy: PersonFilter | None = None,
        update_inplace: bool = False,
    ) -> tuple[pd.DataFrame, dict[int, int], dict]:
        """Compute screning data df from exams and person df.

        Returns:
          The observation df
          pid to row mapping
          A metadata dict about filters etc
        """
        if filter_strategy is None:
            filter_strategy = AtLeastNNonHPV(min_n=2)
        observation_data_transformer = ObservationMatrix()
        included_pids = filter_strategy.filter(self.person_df)
        screening_data: pd.DataFrame = observation_data_transformer.fit_transform(
            self.exams_df[self.exams_df["PID"].isin(included_pids)]
        )
        pid_to_row = observation_data_transformer.pid_to_row
        metadata = {"screenings_filters": filter_strategy.metadata()}

        if update_inplace:
            self.screening_data = screening_data
            self.pid_to_row = pid_to_row
            self.metadata |= metadata
        return (
            screening_data,
            pid_to_row,
            metadata,
        )

    def shape(self) -> tuple[int, int]:
        if self.screening_data is None:
            raise ValueError("Screening data is None!")

        n_rows = self.screening_data["row"].max() + 1
        n_cols = self.screening_data["bin"].max() + 1
        return (n_rows, n_cols)

    def data_as_coo_array(self) -> coo_array:
        if self.screening_data is None:
            raise ValueError("Screening data is None!")

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
        if self.pid_to_row is None:
            raise ValueError("pid_to_row is None! Generate screening data first!")
        cols = cols or ["has_positive", "has_negative", "has_hr", "has_hr_2"]

        n_rows = self.shape()[0]
        n_cols = len(cols)

        # People included in the screening data filtering
        people_in_data = self.person_df[
            self.person_df.index.isin(self.pid_to_row.keys())
        ]
        features = (
            people_in_data.reset_index().melt(id_vars="PID", value_vars=cols).dropna()
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
