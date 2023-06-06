from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

from decipher.exam_data import Diagnosis, ExamTypes, risk_mapping


class PandasTransformerMixin(TransformerMixin):
    """Transformer mixin with type hint set to Pandas."""

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params
    ) -> pd.DataFrame:
        return super().fit_transform(X, y, **fit_params)  # type: ignore


class CleanData(BaseEstimator, PandasTransformerMixin):
    """Check that data has the expected columns on the correct data type"""

    def __init__(self, dtypes: dict[str, Any] | None = None) -> None:
        self.dtypes = dtypes or {
            "cytMorfologi": "category",
            "histMorfologi": "Int64",
            "hpvResultat": "category",
        }
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        required_columns = set(self.dtypes.keys())
        existing_columns = set(X.columns)
        # Check all columns present
        if not required_columns <= existing_columns:
            raise ValueError(
                f"{required_columns - existing_columns} columns not found!"
            )
        # Check correct types
        for column in required_columns:
            if (expected_type := self.dtypes[column]) != (
                actual_type := X.dtypes[column]
            ) and expected_type is not None:
                raise ValueError(
                    f"Column {column} must have dtype {expected_type}, but it is {actual_type}"
                )
            logger.debug(
                f"Column {column} has dtype {actual_type}, expected {expected_type}. {actual_type == expected_type}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class BirthdateAdder(BaseEstimator, PandasTransformerMixin):
    """Adds a birthdate column to the screening data by using the PID mapping to another
    file containing the birth registry.

    The only valid person status is 'B' meaning 'bosatt', other statuses such as 'dead', 'emigrated', etc.
    are not included and will have a None value in the birthdate column.
    """

    dob_map: dict[int, str] = dict()

    def __init__(self, birthday_file: Path, drop_missing: bool = False) -> None:
        self.birthday_file = birthday_file
        self.drop_missing = drop_missing
        """Drop people with missing birth date"""
        self.dob_data: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y=None):
        self.dob_data = (
            (
                pd.read_csv(
                    self.birthday_file,
                    index_col="PID",
                    usecols=["PID", "FOEDT"],
                    parse_dates=["FOEDT"],
                    dayfirst=True,
                )["FOEDT"]
            )
            if self.dob_data is None
            else self.dob_data
        )
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.join(self.dob_data, on="PID")
        if X["FOEDT"].isna().any():
            if not self.drop_missing:
                raise ValueError("Some people are missing birth info!")
            number_missing_birth_data = X[X["FOEDT"].isna()]["PID"].nunique()
            logger.warning(
                f"{number_missing_birth_data} people are missing birth info!"
            )
            X = X.dropna(subset="FOEDT")
        return X


class ToExam(BaseEstimator, PandasTransformerMixin):
    def __init__(self, fields_to_keep: list | None = None) -> None:
        self.fields_to_keep = fields_to_keep or ["PID", "FOEDT"]
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mapper = {
            "cytDate": "cytMorfologi",
            "histDate": "histMorfologi",
            "hpvDate": "hpvResultat",
        }

        # Transform from wide to long
        exams = (
            X.reset_index()
            .melt(
                id_vars="index",
                value_vars=mapper.keys(),  # type: ignore[arg-type]
                var_name="exam_type",
                value_name="exam_date",
            )
            .dropna()
            .astype({"exam_type": "category"})
        )

        # Join on result columns
        exams = exams.join(X[mapper.values()], on="index")  # type: ignore[call-overload]

        # Add result column
        conditions = [exams["exam_type"] == key for key in mapper]
        values = [exams[key] for key in mapper.values()]
        exams["exam_diagnosis"] = np.select(conditions, values)

        # Drop the raw exam result
        exams = exams.drop(columns=mapper.values())

        # Remap exam types
        exams["exam_type"] = exams["exam_type"].transform(self._map_exam_type)
        exams["exam_diagnosis"] = (
            exams["exam_diagnosis"]
            .astype("str")
            .apply(lambda diagnosis_string: Diagnosis(diagnosis_string))
            .astype("category")
        )

        return exams.join(X[self.fields_to_keep], on="index")

    @staticmethod
    def _map_exam_type(field_name) -> ExamTypes:
        return {
            "cytDate": ExamTypes.Cytology,
            "histDate": ExamTypes.Histology,
            "hpvDate": ExamTypes.HPV,
        }[field_name]


class AgeAdder(BaseEstimator, PandasTransformerMixin):
    def __init__(
        self, date_field: str, birth_field: str, age_field: str = "age"
    ) -> None:
        self.date_field = date_field
        self.birth_field = birth_field
        self.age_field = age_field
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.age_field] = X[self.date_field] - X[self.birth_field]
        return X


class RiskAdder(BaseEstimator, PandasTransformerMixin):
    def fit(self, X, y=None):
        if "exam_diagnosis" not in X:
            raise ValueError("No exam diagnosis! Is this an exam DF?")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["risk"] = (
            X["exam_diagnosis"]
            .transform(lambda diagnosis: risk_mapping[diagnosis])
            .astype("Int64")
        )
        return X


@dataclass
class PersonFeature:
    """Helper class for PersonStats"""

    name: str
    initial_value: Any
    getter: Callable


class PersonStats(BaseEstimator, PandasTransformerMixin):
    """Take an exam DF, and generate stats per person"""

    def __init__(self, base_df: pd.DataFrame | None = None) -> None:
        self.base_df = base_df
        self.high_risk_hpv_types: list[str | int] = [
            16,
            8,
            45,
            31,
            33,
            35,
            52,
            58,
            39,
            51,
            56,
            59,
            68,
            "HR",
            "HF",
            "HD",
            "HE",
        ]
        self.low_risk_hpv_types: list[str | int] = [11, 6]

    def fit(self, X: pd.DataFrame, y=None):
        CleanData(
            dtypes={
                "PID": "int64",
                "exam_type": None,
                "exam_date": "datetime64[ns]",
                "age": "timedelta64[ns]",
            }
        ).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        person_df = X.groupby("PID")[["age", "risk"]].agg(["min", "max", "mean"])
        person_df.columns = [
            "_".join(column) for column in person_df.columns
        ]  # type: ignore # Flatten columns
        person_df = person_df.join(X.groupby("PID")["FOEDT"].agg("first"), on="PID")

        if self.base_df is not None:
            person_df = person_df.join(self._get_hpv_features(X, person_df))
        return person_df

    def _get_hpv_features(
        self, exams_df: pd.DataFrame, person_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Construct and return a DataFrame with a variety of HPV related features.

        The features include:
        - 'has_positive': Count of positive HPV results per PID.
        - 'has_negative': Count of negative HPV results per PID.
        - 'number_of_screenings': Count of screenings per PID.
        - 'age_last_exam': Age at last exam per PID.
        - 'hr_count': Count of high-risk HPV types per PID.
        - 'lr_count': Count of low-risk HPV types per PID.
        - 'age_first_hr': Age at first detection of high-risk HPV types per PID.
        - 'age_first_lr': Age at first detection of low-risk HPV types per PID.
        - 'age_first_positive': Age at first positive HPV result per PID.
        - 'age_first_negative': Age at first negative HPV result per PID.

        Arguments:
          - exams_df - Should have 'PID', 'risk', 'age' and 'exam_diagnosis' fields.
          - person_df - Should have 'PID' and 'FOEDT' fields.
        """
        if self.base_df is None:
            raise ValueError()
        hpv_details_df: pd.DataFrame = HPVResults().fit_transform(self.base_df)

        feature_df = pd.DataFrame(index=self.base_df["PID"].unique())

        def _count_where_result_match(match: str, field_to_query: str = "hpvResultat"):
            # MyPy does not infer the correct type of self.base_df inside the closure
            # even though we have a guard above.
            counts = (
                cast(pd.DataFrame, self.base_df)
                .query(f"{field_to_query} == '{match}'")["PID"]
                .value_counts()
            )
            return counts

        def age_first_field_match(match: str, field_to_query: str = "exam_diagnosis"):
            ages = (
                exams_df.query(f"{field_to_query} == '{match}'")
                .groupby("PID")["age"]
                .agg("min")
                .apply(lambda x: x.days / 365)
            )
            return ages

        def _hpv_type_counts(hr_types: list[str | int]):
            """Get pid-counts for hr_types"""
            pid_counts = hpv_details_df[hpv_details_df["value"].isin(hr_types)][
                "PID"
            ].value_counts()
            return pid_counts

        def _age_first_hr(
            hr_types: list[str | int],
        ):
            """Get pid-age for hr_types"""
            dates = (
                hpv_details_df[hpv_details_df["value"].isin(hr_types)]
                .groupby("PID")["hpvDate"]
                .agg("min")
            )
            ages = (dates - person_df["FOEDT"]).apply(lambda x: x.days / 365)
            return ages

        feature_df["count_positive"] = _count_where_result_match(
            match="positiv"
        ).reindex(feature_df.index, fill_value=0)
        feature_df["count_negative"] = _count_where_result_match(
            match="negativ"
        ).reindex(feature_df.index, fill_value=0)
        feature_df["number_of_screenings"] = exams_df.dropna(subset=["risk"])[
            "PID"
        ].value_counts()
        feature_df["age_last_exam"] = (
            exams_df.groupby("PID")["age"].agg("max").apply(lambda x: x.days / 365)
        )
        feature_df["hr_count"] = _hpv_type_counts(
            hr_types=self.high_risk_hpv_types
        ).reindex(feature_df.index, fill_value=0)
        feature_df["lr_count"] = _hpv_type_counts(
            hr_types=self.low_risk_hpv_types
        ).reindex(feature_df.index, fill_value=0)
        feature_df["age_first_hr"] = _age_first_hr(hr_types=self.high_risk_hpv_types)
        feature_df["age_first_lr"] = _age_first_hr(hr_types=self.low_risk_hpv_types)
        feature_df["age_first_positive"] = age_first_field_match(match="positiv")
        feature_df["age_first_negative"] = age_first_field_match(match="negativ")

        return feature_df


class HPVResults(BaseEstimator, PandasTransformerMixin):
    """Take a raw DF, and generate HPV results

    Warning:
      HPV negative and hpv non-conclusive are _not_ included!!!
    """

    def fit(self, X: pd.DataFrame, y=None):
        CleanData(
            dtypes={"PID": "int64", "hpvTesttype": None, "hpvDate": "datetime64[ns]"}
        ).fit(X)
        self.hpv_genotype_columns = [
            col for col in X.columns if col.endswith("Genotype")
        ]
        if not self.hpv_genotype_columns:
            raise ValueError("There are no Genotype columns!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.dropna(subset="hpvDate")
            .reset_index(names="exam_index")
            .melt(
                id_vars=["PID", "exam_index", "hpvTesttype", "hpvDate"],
                value_vars=self.hpv_genotype_columns,
            )
            .dropna(subset="value")
        ).astype({"variable": "category", "value": "category"})


class ObservationMatrix(BaseEstimator, PandasTransformerMixin):
    """Convert exams df to observations"""

    def __init__(
        self, risk_agg_method: str | Callable = "max", months_per_bin: float = 3
    ):
        self.risk_agg_method = risk_agg_method
        self.months_per_bin = months_per_bin
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        CleanData(dtypes={"PID": "int64", "age": "timedelta64[ns]", "risk": "Int64"})
        # Make the time bins
        days_per_month = 30
        bin_width = timedelta(days=self.months_per_bin * days_per_month)
        self.bins: npt.NDArray = np.arange(
            X["age"].min(),
            X["age"].max() + bin_width,  # Add to ensure endpoint is included
            bin_width,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "PID": X["PID"],
                "risk": X["risk"],
                "bin": pd.cut(
                    X["age"],
                    self.bins,
                    right=True,
                    labels=False,
                    include_lowest=True,
                ),  # type: ignore[call-overload]  # right=False indicates close left side
            }
        )
        assert not out[["PID", "bin"]].isna().any().any(), "You fucked up"
        out = out.dropna()  # Drop nan risk

        # The observed=True is important!
        # As the bin is categorical, observed=False will produce a cartesian product
        # between all possible bins and all rows. This eats up a lot of memory!
        return out.groupby(["PID", "bin"], as_index=False, observed=True)["risk"].agg(
            self.risk_agg_method
        )  # type: ignore[return-value]  # as_index=False makes this a DataFrame
