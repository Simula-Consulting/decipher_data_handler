import functools
import json
import operator
from abc import ABC, abstractmethod
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from decipher.processing.pipeline import get_base_pipeline, read_raw_df
from decipher.processing.transformers import (
    AgeAdder,
    ObservationMatrix,
    PersonStats,
    RiskAdder,
    ToExam,
)

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
    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
        ...

    def metadata(self) -> dict:
        return {"type": "generic_filter"}


class MaximumTimeSeparation(PersonFilter):
    """Filter people where the time between last two exams is less than `max_time`.

    This is useful for assuring there is sufficiently little time the last non-masked
    exam and the exam we attempt to predict for.

    People with less than two screenings are disregarded.

    Arguments:
        max_time: Maximum time between last two exams, in number of bins.

    Warning:
        Do note that this will tend to bias our selection, as time between exams is not
        uniformly distributed.
    """

    def __init__(self, max_time_difference: int) -> None:
        self.max_time_difference = max_time_difference

    @classmethod
    def from_time_delta(cls, max_time_difference: timedelta, days_per_bin: int):
        """Create a MaximumTimeSeparation from a timedelta.

        Requires days_per_bin to be known.
        By default, the days_per_bin is 30 (days per month) * 3 (months per bin) = 90.
        See `decipher.processing.transformers.ObservationMatrix` for more information.
        """
        return cls(int(max_time_difference / timedelta(days=days_per_bin)))

    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
        number_of_screenings = screening_df["PID"].value_counts()
        more_than_one = number_of_screenings[number_of_screenings > 1].index
        time_difference = (
            screening_df[screening_df["PID"].isin(more_than_one)]
            .sort_values("bin", ascending=False)
            .groupby("PID")["bin"]
            .agg(lambda ages: ages.iloc[0] - ages.iloc[1])
        )
        return time_difference[time_difference <= self.max_time_difference].index

    def metadata(self) -> dict:
        return {
            "type": "max_time_difference",
            "max_time_difference": str(self.max_time_difference),
        }


class AtLeastNNonHPV(PersonFilter):
    def __init__(self, min_n: int = 2):
        self.min_n = min_n

    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
        exam_count_per_person = screening_df["PID"].value_counts()
        return exam_count_per_person[exam_count_per_person >= self.min_n].index

    def metadata(self) -> dict:
        return {"type": "min_n_non_hpv", "min_n": self.min_n}


class TrueFields(PersonFilter):
    """Filter people where all fields in `fields` are True."""

    def __init__(self, fields: list[str]):
        self.fields = fields

    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
        matches = person_df[self.fields].all(axis=1)
        return person_df[matches].index

    def metadata(self) -> dict:
        return {"type": "true fields", "fields": self.fields}


class SaneAgeFilter(PersonFilter):
    """Filter out people with unreasonable exam ages.

    For people where the age of the exam is either very young or very old, remove them.
    """

    def __init__(self, min_age: timedelta, max_age: timedelta):
        self.min_age = min_age
        self.max_age = max_age

    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
        return person_df[
            (person_df["age_min"] >= self.min_age)
            & (person_df["age_max"] <= self.max_age)
        ].index

    def metadata(self) -> dict:
        return {
            "type": "sane_age_filter",
            "min_age": str(self.min_age),
            "max_age": str(self.max_age),
        }


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

    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
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
        if not filters:
            raise ValueError("Filters cannot be empty!")
        self.filters = filters

    def filter(
        self, person_df: pd.DataFrame, screening_df: pd.DataFrame
    ) -> Iterable[int] | "pd.Series[int]":
        return functools.reduce(
            operator.and_,
            (set(filter_.filter(person_df, screening_df)) for filter_ in self.filters),
        )

    def metadata(self) -> dict:
        return {
            "type": "combine filter",
            "sub filters": [filter_.metadata() for filter_ in self.filters],
        }


class DataManager:
    def __init__(
        self,
        person_df: pd.DataFrame,
        exams_df: pd.DataFrame,
        screening_data: pd.DataFrame | None = None,
        metadata: dict | None = None,
    ):
        self.screening_data = screening_data
        self.person_df = person_df
        self.exams_df = exams_df
        self.metadata = metadata or {"decipher_version": version("decipher")}

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

    def save_to_parquet(
        self, directory: Path, engine: _parquet_engine_types = "auto"
    ) -> None:
        if not directory.is_dir():
            raise ValueError()
        if self.screening_data is not None:
            self.screening_data.to_parquet(
                directory / "screening_data.parquet", engine=engine
            )
        self.person_df.to_parquet(directory / "person_df.parquet", engine=engine)
        self.exams_df.to_parquet(directory / "exams_df.parquet", engine=engine)
        with open(directory / "metadata.json", "w") as file:
            # We always want to store the decipher version, so if it is not
            # in the metadata, add it.
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
        else:
            screening_data = None
        person_df = pd.read_parquet(directory / "person_df.parquet", engine=engine)
        exams_df = pd.read_parquet(directory / "exams_df.parquet", engine=engine)
        return DataManager(
            person_df=person_df,
            exams_df=exams_df,
            screening_data=screening_data,
            metadata=metadata,
        )

    def get_screening_data(
        self,
        filter_strategy: PersonFilter | None = None,
        update_inplace: bool = False,
        months_per_bin: int = 3,
    ) -> tuple[pd.DataFrame, dict]:
        """Compute screening data df from exams and person df.

        The output DataFrame will have the following columns:
          - PID: The person ID
          - bin: The bin number
          - risk: The risk of the bin
          - is_last: Whether the bin is the last bin for the person

        Returns:
          The observation df
          A metadata dict about filters etc

        Examples:
            To convert to an array, you may for example use coo array
            >>> from scipy.sparse import coo_array
            >>> coo_array(
            ...     (
            ...         data_manager.screening_data["risk"],
            ...         (
            ...             data_manager.screening_data["PID"].map(pid_to_row),
            ...             data_manager.screening_data["bin"],
            ...         ),
            ...     ),
            ...     shape=(len(pid_to_row), data_manager.screening_data["bin"].max() + 1),
            ...     dtype="int8",
            ... )

            Or more directly, something like
            >>> rows = data_manager.screening_data["PID"].map(pid_to_row)
            >>> cols = data_manager.screening_data["bin"]
            >>> observed = np.zeros((rows.max() + 1, cols.max() + 1), dtype="int8")
            >>> observed[row, col] = data_manager.screening_data["risk"]
        """
        if filter_strategy is None:
            filter_strategy = AtLeastNNonHPV(min_n=2)

        observation_data_transformer = ObservationMatrix(months_per_bin=months_per_bin)
        screening_data: pd.DataFrame = observation_data_transformer.fit_transform(
            self.exams_df
        )
        included_pids = filter_strategy.filter(self.person_df, screening_data)
        screening_data = screening_data[screening_data["PID"].isin(included_pids)]

        # Add a new column, which indicates the last bin per person (PID)
        logger.debug("Adding is_last column")
        screening_data["is_last"] = (
            screening_data.groupby("PID")["bin"].transform("max")
            == screening_data["bin"]
        )

        assert not screening_data.isna().values.any()
        assert screening_data["risk"].isin(range(1, 5)).all()
        metadata = {
            "screenings_filters": filter_strategy.metadata(),
            "bins": self._format_screening_bins(observation_data_transformer.bins),
        }

        if update_inplace:
            self.screening_data = screening_data
            self.metadata |= metadata
        return (
            screening_data,
            metadata,
        )

    @staticmethod
    def _format_screening_bins(bins) -> list[float]:
        """Format the bins of observation_data_transformer to be suitable for the metadata."""

        def timedelta_to_years(timedelta) -> float:
            return timedelta.days / 365

        return [round(years, 3) for years in map(timedelta_to_years, bins.tolist())]

    def get_last_screening_bin(self) -> dict[int, int]:
        """Get the bin of the last screening per person."""
        if self.screening_data is None:
            raise ValueError("Screening data is None!")

        return dict(self.screening_data.groupby("PID")["bin"].agg("max"))

    def get_feature_data(
        self, columns: Iterable[str] | None = None, pids: Iterable | None = None
    ) -> pd.DataFrame:
        """Get the feature data for the given columns and pids.

        Args:
            columns: The columns to get. If None, will use the default columns.
            pids: The pids to get. If None, will get all pids.

        Example:
            >>> features = data_manager.get_feature_data(pids=pid_to_row)
            >>> feature_array = coo_array(
            ...     (
            ...         features["value"],
            ...         (features["PID"].map(pid_to_row), features["feature"].map(feature_to_column)),
            ...     ),
            ...     shape=(len(pid_to_row), len(variable_to_column)),
            ...     dtype="int8",
            ... )
        """
        columns = (
            list(columns)
            if columns is not None
            else [
                "has_positive",
                "has_negative",
                "has_hr",
                "has_hr_2",
            ]
        )
        if not set(columns).issubset(self.person_df.columns):
            raise ValueError(
                f"{set(columns) - set(self.person_df.columns)} are not in the person_df"
            )

        people_in_data = (
            self.person_df[self.person_df.index.isin(pids)]
            if pids is not None
            else self.person_df
        )
        return (
            people_in_data.reset_index()
            .melt(id_vars="PID", value_vars=columns, var_name="feature")
            .dropna()
        )
