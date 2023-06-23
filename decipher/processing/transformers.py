from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

from decipher.exam_data import HPV_TEST_TYPE_NAMES, Diagnosis, ExamTypes, risk_mapping


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


def timedelta_to_years(timedeltas: pd.Series) -> pd.Series:
    return timedeltas.apply(lambda x: x.days / 365)


class PersonStats(BaseEstimator, PandasTransformerMixin):
    """Take an exam DF, and generate stats per person"""

    def __init__(self, base_df: pd.DataFrame | None = None) -> None:
        self.base_df = base_df

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
        # TODO: Should HPV results be included in the min/max/mean of the age?
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
        - 'count_positive': Count of positive HPV results.
        - 'count_negative': Count of negative HPV results.
        - 'number_of_screenings': Count of screenings.
        - 'age_last_exam': Age at last exam.
        - 'age_first_positive': Age at first positive HPV result.
        - 'age_first_negative': Age at first negative HPV result.
        - 'count_positive_last_5_years': Count of positive HPV results in the last 5 years before the last exam.
        - 'count_negative_last_5_years': Count of negative HPV results in the last 5 years before the last exam.

        Arguments:
          - exams_df - Should have 'PID', 'risk', 'age' and 'exam_diagnosis' fields.
          - person_df - Should have 'PID' and 'FOEDT' fields.
        """
        if self.base_df is None:
            raise ValueError()

        feature_df = pd.DataFrame(index=self.base_df["PID"].unique())

        feature_df["count_positive"] = (
            self.base_df.query("hpvResultat == 'positiv'")["PID"]
            .value_counts()
            .reindex(feature_df.index, fill_value=0)
        )
        feature_df["count_negative"] = (
            self.base_df.query("hpvResultat == 'negativ'")["PID"]
            .value_counts()
            .reindex(feature_df.index, fill_value=0)
        )
        feature_df["number_of_screenings"] = (
            exams_df.dropna(subset=["risk"])["PID"]
            .value_counts()
            .reindex(feature_df.index, fill_value=0)
        )
        # person_df.age_max contains the age of the last exam of the entire
        # exams_df. This also includes HPV results, which we don't want.
        feature_df["age_last_exam"] = timedelta_to_years(
            exams_df.query("exam_type != 'HPV'").groupby("PID")["age"].agg("max")
        )
        birth_date = person_df["FOEDT"]
        feature_df["age_first_positive"] = self.datetime_to_age(
            self.base_df.query("hpvResultat == 'positiv'")
            .groupby("PID")["hpvDate"]
            .agg("min"),
            birth_date,
        )
        feature_df["age_first_negative"] = self.datetime_to_age(
            self.base_df.query("hpvResultat == 'negativ'")
            .groupby("PID")["hpvDate"]
            .agg("min"),
            birth_date,
        )

        times_of_last_exam = person_df["age_max"] + person_df["FOEDT"]
        five_years = timedelta(days=5 * 365)
        feature_df["count_positive_last_5_years"] = self.count_in_time_window(
            self.base_df.query("hpvResultat == 'positiv'")[["hpvDate", "PID"]].rename(
                columns={"hpvDate": "time"}
            ),
            times_of_last_exam,
            five_years,
        ).reindex(feature_df.index, fill_value=0)
        feature_df["count_negative_last_5_years"] = self.count_in_time_window(
            self.base_df.query("hpvResultat == 'negativ'")[["hpvDate", "PID"]].rename(
                columns={"hpvDate": "time"}
            ),
            times_of_last_exam,
            five_years,
        ).reindex(feature_df.index, fill_value=0)

        return feature_df

    @staticmethod
    def datetime_to_age(
        times: pd.Series,
        birth_dates: pd.Series,
    ) -> pd.Series:
        """Convert a series of datetimes to a series of ages, as year in float.

        Args:
            times: the times to convert
            birth_dates: the birth dates to use for the conversion

        Returns:
            Series with the ages, as year in float.
        """
        return timedelta_to_years(times - birth_dates)

    @staticmethod
    def count_in_time_window(
        times: pd.DataFrame,
        last_date: pd.Series,
        time_window: timedelta,
    ) -> pd.Series:
        """Count the number of occurrences per PID inside a time window.

        The time window is counted backwards from the last_date. I.e. if the last_date
        is 2020-01-10, and the time_window is 5 days, then the time window is 2020-01-05
        to 2020-01-10.

        Args:
            times: the times to count. Should have the columns "PID" and "time"
            last_date: the last date for each person. PID as index.
            time_window: the time window to count in, counted backwards from last_date

        Returns:
            Series with the counts per PID
        """
        if not times["PID"].isin(last_date.index).all():
            raise ValueError("Some PIDs from times are not in last_dates")
        start_times = times["PID"].map(last_date - time_window)  # type: ignore[operator]  # Series is type datetime, but this can't be typed.
        end_times = times["PID"].map(last_date)
        return times[times["time"].between(start_times, end_times)][
            "PID"
        ].value_counts()


class HPVResults(BaseEstimator, PandasTransformerMixin):
    """Take a raw DF, and generate HPV results

    The resulting DF will have the following columns:
    - PID
    - exam_index: the index of the exam in the raw data
    - hpvTesttype
    - hpvDate
    - genotype_field: the genotype column of the raw data, i.e. hpv1Genotype, hpv2Genotype, etc.
    - genotype: the genotype, i.e. 16, 18, HR, etc
    - hpv_test_type_name: the name of the test type, i.e. "Cobas 4800 System".

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
        hpv_df = (
            X.dropna(subset="hpvDate")
            .reset_index(names="exam_index")
            .melt(
                id_vars=["PID", "exam_index", "hpvTesttype", "hpvDate"],
                value_vars=self.hpv_genotype_columns,
                var_name="genotype_field",
                value_name="genotype",
            )
            .dropna(subset="genotype")
        ).astype(
            {"genotype_field": "category", "genotype": "category", "hpvTesttype": "int"}
        )
        hpv_df["test_type_name"] = (
            hpv_df["hpvTesttype"].map(HPV_TEST_TYPE_NAMES).astype("category")
        )
        return hpv_df


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
