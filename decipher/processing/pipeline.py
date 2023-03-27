import logging
from functools import partial
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from decipher.processing.transformers import (
    AgeAdder,
    BirthdateAdder,
    CleanData,
    HPVResults,
    ObservationMatrix,
    RiskAdder,
    ToExam,
)

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
        "histTopografi": "category",  # We don't use this column
    } | {f"hpv{i}Genotype": "category" for i in range(1, 6)}
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
    metadata = {"decipher_version": version("decipher")} | metadata
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


def get_base_pipeline(
    birthday_file: Path, drop_missing_birthday: bool = False
) -> Pipeline:
    """Base pipeline for reading from raw"""
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
        ]
    )


def get_exam_pipeline(
    base_pipeline: Pipeline | None = None,
    birthday_file: Path | None = None,
    drop_missing_birthday: bool = False,
) -> Pipeline:
    if base_pipeline is None:
        if birthday_file is None:
            raise ValueError(
                "You must supply a birthday_file when base_pipeline is None"
            )
        base_pipeline = get_base_pipeline(
            birthday_file=birthday_file, drop_missing_birthday=drop_missing_birthday
        )
    elif birthday_file is not None:
        logger.warning(
            "A pipeline and birthday_file was supplied."
            "The birthday file will be ignored."
        )
    return Pipeline(
        [
            ("base_pipeline", base_pipeline),
            ("wide_to_long", ToExam()),
            ("age_adder", AgeAdder(date_field="exam_date", birth_field="FOEDT")),
            ("risk_mapper", RiskAdder()),
        ],
        verbose=True,
    )


def get_hpv_pipeline(base_pipeline: Pipeline) -> Pipeline:
    return Pipeline(
        [
            ("base_pipeline", base_pipeline),
            ("hpv_results", HPVResults()),
        ]
    )


def get_observations_pipeline(
    exam_pipeline: Pipeline | None = None,
    birthday_file: Path | None = None,
    drop_missing_birthday: bool = False,
    risk_agg_method: str = "max",
    months_per_bin: float = 3,
) -> Pipeline:
    if exam_pipeline is None:
        if birthday_file is None:
            raise ValueError(
                "You must supply a birthday_file when base_pipeline is None"
            )
        exam_pipeline = get_exam_pipeline(
            birthday_file=birthday_file, drop_missing_birthday=drop_missing_birthday
        )
    elif birthday_file is not None:
        logger.warning(
            "A pipeline and birthday_file was supplied."
            "The birthday file will be ignored."
        )
    return Pipeline(
        [
            ("exam_pipeline", exam_pipeline),
            (
                "observations",
                ObservationMatrix(
                    months_per_bin=months_per_bin, risk_agg_method=risk_agg_method
                ),
            ),
        ],
        verbose=True,
    )
