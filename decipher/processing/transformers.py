import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from decipher.exam_data import Diagnosis, ExamTypes, risk_mapping

logger = logging.getLogger(__name__)


class CleanData(BaseEstimator, TransformerMixin):
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

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class BirthdateAdder(BaseEstimator, TransformerMixin):
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
        self.dob_data = self.dob_data or (
            pd.read_csv(
                self.birthday_file,
                dtype={"STATUS": "category"},
                parse_dates=["FOEDT"],
                dayfirst=True,
            )
            .query("STATUS == 'B'")  # Take only rows giving birth data
            .set_index("PID")["FOEDT"]
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


class ToExam(BaseEstimator, TransformerMixin):
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


class AgeAdder(BaseEstimator, TransformerMixin):
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


class RiskAdder(BaseEstimator, TransformerMixin):
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


class PersonStats(BaseEstimator, TransformerMixin):
    """Take an exam DF, and generate stats per person"""

    def fit(self, X: pd.DataFrame, y=None):
        CleanData(
            dtypes={
                "PID": "int",
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
        person_df = person_df.join(X.groupby("PID").agg("first"), on="PID")
        count_per_person_per_exam_type = (
            X.groupby("exam_type", as_index=False)["PID"]  # type: ignore[operator]
            .value_counts()
            .pivot(index="PID", columns="exam_type", values="count")
        )
        count_per_person_per_exam_type.columns = [
            f"{col}_count" for col in count_per_person_per_exam_type.columns
        ]
        person_df = person_df.join(count_per_person_per_exam_type, on="PID")
        return person_df


class HPVResults(BaseEstimator, TransformerMixin):
    """Take an exam DF, and generate HPV results"""

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
            .dropna()
        )
