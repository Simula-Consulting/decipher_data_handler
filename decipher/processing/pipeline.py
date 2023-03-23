import logging
from functools import partial
from pathlib import Path

import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from decipher.processing.transformers import AgeAdder, BirthdateAdder, CleanData, ToExam

logger = logging.getLogger(__name__)


def read_raw_df(
    screening_data_path: Path,
    dtypes: dict | None = None,
    datetime_cols: list | None = None,
    **read_csv_kwargs,
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
        comment="#",
        **read_csv_kwargs,
    )


def read_from_csv(path: Path, df_reader=None) -> tuple[pd.DataFrame, dict]:
    """Read df from a csv, with header yaml"""
    if not path.exists():
        raise ValueError(f"{path} does not exist!")
    if df_reader is None:
        df_reader = partial(pd.read_csv, comment="#")
    header = ""
    with open(path) as file:
        while (line := file.readline()).startswith("#"):
            header += line.strip("#")[1:]
    print(header, "-------------")
    with open(path) as file:
        print(file.read())
    header_data = yaml.safe_load(header)
    data = df_reader(path)
    return data, header_data


def write_to_csv(path: Path, df: pd.DataFrame, metadata: dict, **pd_kwargs) -> None:
    if path.exists():
        logger.warning(f"{path} exists, overwriting!")
    metadata_yaml = yaml.dump(metadata).split("\n")
    with open(path, "w") as file:
        # Add a '# ' to the beginning of each line
        file.write(
            "\n".join(
                f"# {line}" for line in metadata_yaml if line
            )  # Conditional to remove empty lines
        )
        file.write("\n")
    df.to_csv(path, mode="a", **pd_kwargs)


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
