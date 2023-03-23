from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from decipher.processing.transformers import AgeAdder, BirthdateAdder, CleanData, ToExam


def read_raw_df(
    screening_data_path: Path,
    dtypes: dict | None = None,
    datetime_cols: list | None = None,
) -> pd.DataFrame:
    """Read raw screening data into DF"""
    if not screening_data_path.exists():
        raise ValueError(f"{screening_data_path} does not exist!")
    dtypes = dtypes or {
        "cytMorfologi": "category",
        "histMorfologi": "Int64",
        "hpvResultat": "category",
    }
    datetime_cols = datetime_cols or ["hpvDate", "cytDate", "histDate"]

    return pd.read_csv(
        screening_data_path,
        dtype=dtypes,
        parse_dates=datetime_cols,
        dayfirst=True,
    )


def exam_pipeline(birthday_file: Path, drop_missing_birthday: bool = False) -> Pipeline:
    if not birthday_file.exists():
        raise ValueError(f"{birthday_file} does not exist!")
    return Pipeline(
        [
            ("cleaner", CleanData()),
            (
                "birthdate_adder",
                BirthdateAdder(
                    birthday_file=birthday_file, drop_missing=drop_missing_birthday
                ),
            ),
            ("wide_to_long", ToExam()),
            ("age_adder", AgeAdder(date_field="exam_date", birth_field="FOEDT")),
        ],
        verbose=True,
    )
