import logging
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pytest

from decipher.processing.pipeline import (
    get_base_pipeline,
    get_exam_pipeline,
    get_hpv_pipeline,
    read_from_csv,
    read_raw_df,
    write_to_csv,
)
from decipher.processing.transformers import HPVResults, PersonStats

logger = logging.getLogger(__name__)

test_data_base = Path(__file__).parent / "test_processing_datasets"
test_data_screening = test_data_base / "test_screening_data.csv"
test_data_dob = test_data_base / "test_dob_data.csv"


def test_read_and_exam_pipeline():
    """Simply try reading and running pipeline"""
    raw = read_raw_df(test_data_screening)

    # This should fail as we have people with missing birth data
    with pytest.raises(ValueError):
        exam_pipeline = get_exam_pipeline(
            birthday_file=test_data_dob, drop_missing_birthday=False
        )
        exam_pipeline.fit_transform(raw)

    exam_pipeline = get_exam_pipeline(
        birthday_file=test_data_dob, drop_missing_birthday=True
    )
    exam_df = exam_pipeline.fit_transform(raw)
    logger.debug(exam_df)


def test_read_and_hpv_pipeline():
    """Simply try reading and running pipeline"""
    raw = read_raw_df(test_data_screening)

    base_pipeline = get_base_pipeline(
        birthday_file=test_data_dob, drop_missing_birthday=True
    )
    hpv_pipeline = get_hpv_pipeline(base_pipeline=base_pipeline)
    hpv_df = hpv_pipeline.fit_transform(raw)
    logger.debug(hpv_df)


def test_person_stats():
    raw = read_raw_df(test_data_screening)

    pipeline = get_exam_pipeline(
        birthday_file=test_data_dob, drop_missing_birthday=True
    )
    exams = pipeline.fit_transform(raw)

    person_df = PersonStats().fit_transform(exams)
    logger.debug(f"Person df:\n {person_df}")
    logger.debug(f"Person df columns: {list(person_df.columns)}")

    expected_columns = {
        "age_min",
        "age_max",
        "age_mean",
        "risk_min",
        "risk_max",
        "risk_mean",
        "FOEDT",
        "cytology_count",
        "histology_count",
        "HPV_count",
    }
    assert set(person_df.columns) == expected_columns


def test_hpv_results():
    raw = read_raw_df(test_data_screening)
    hpv_df = HPVResults().fit_transform(raw)
    logger.debug(f"HPV DF:\n{hpv_df.head()}")


def test_read_from_csv(tmp_path: Path):
    data_file = tmp_path / "data.csv"
    with open(data_file, "w") as file:
        file.write(
            "# key1: value1\n"
            "# key2: value2\n"
            "# list:\n"
            "#   - one\n"
            "#   - two\n"
            "col1,col2\n"
            "a,b\n"
        )
    data, metadata = read_from_csv(data_file)
    assert metadata == {
        "key1": "value1",
        "key2": "value2",
        "list": ["one", "two"],
    }
    assert pd.DataFrame({"col1": ["a"], "col2": ["b"]}).equals(data)


def test_write_to_csv(tmp_path: Path):
    data_file = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    metadata = {"key": "value", "list": ["foo", "bar"]}
    write_to_csv(data_file, df=df, metadata=metadata)
    with open(data_file) as file:
        logger.debug(f"Current file content: {file.read()}")
    assert read_from_csv(data_file)[1] == metadata | {
        "decipher_version": version("decipher")
    }
