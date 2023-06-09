import logging
from datetime import datetime, timedelta
from importlib.metadata import version
from math import ceil
from pathlib import Path

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis.extra.pandas import column, data_frames, range_indexes

from decipher.exam_data import HPV_TEST_TYPE_NAMES
from decipher.processing.pipeline import (
    get_base_pipeline,
    get_exam_pipeline,
    get_hpv_pipeline,
    read_from_csv,
    read_raw_df,
    write_to_csv,
)
from decipher.processing.transformers import (
    HPVResults,
    ObservationMatrix,
    PersonStats,
    ToExam,
)

logger = logging.getLogger(__name__)

test_data_base = Path(__file__).parent / "test_processing_datasets"
test_data_screening = test_data_base / "test_screening_data.csv"
test_data_dob = test_data_base / "test_dob_data.csv"


def test_base_pipeline():
    raw = read_raw_df(test_data_screening)

    # This should fail as we have people with missing birth data
    with pytest.raises(ValueError):
        base_pipeline = get_base_pipeline(
            birthday_file=test_data_dob, drop_missing_birthday=False
        )
        base_pipeline.fit_transform(raw)

    base_pipeline = get_base_pipeline(
        birthday_file=test_data_dob, drop_missing_birthday=True
    )


def test_to_exam():
    base_df = get_base_pipeline(
        test_data_dob, drop_missing_birthday=True
    ).fit_transform(read_raw_df(test_data_screening))
    exams = ToExam().fit_transform(base_df)
    assert exams["detailed_exam_type"].notna().all()
    assert exams["detailed_exam_type"].dtype == "category"

    assert set(exams["exam_type"]) == {"HPV", "cytology", "histology"}
    assert set(exams["detailed_exam_type"]) == set(HPV_TEST_TYPE_NAMES.values()) | {
        "cytology",
        "histology",
    }


def test_read_and_exam_pipeline():
    """Simply try reading and running pipeline"""
    raw = read_raw_df(test_data_screening)

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


def test_observation_out():
    raw = read_raw_df(test_data_screening)

    exam_pipeline = get_exam_pipeline(
        birthday_file=test_data_dob, drop_missing_birthday=True
    )
    exam_df: pd.DataFrame = exam_pipeline.fit_transform(raw)  # type: ignore
    observations = ObservationMatrix().fit_transform(exam_df)
    logger.info(observations)

    assert {"bin", "PID", "risk"} == set(observations)
    # Assert only one risk per person per time
    assert observations.value_counts(subset=["PID", "bin"]).unique() == [1]

    # Check correct number of bins
    min_age = exam_df["age"].min()
    max_age = exam_df["age"].max()

    days_per_month = 30
    months_per_bin = 3
    number_of_bins = ceil((max_age - min_age).days / days_per_month / months_per_bin)
    assert observations["bin"].max() == number_of_bins - 1  # Compensate for 0-indexing


age16 = timedelta(days=365 * 16)
age60 = timedelta(days=365 * 60)


@given(
    data_frames(
        [
            column(
                "age_days",
                elements=st.integers(min_value=age16.days, max_value=age60.days),
            ),
            column("PID", elements=st.integers(min_value=0)),
            column("risk", elements=st.sampled_from(range(1, 5))),
        ],
        index=range_indexes(min_size=1),
    )
)
def test_observation(exams: pd.DataFrame):
    age_range = exams["age_days"].max() - exams["age_days"].min()
    assume(age_range / 30 / 3 > 1)  # We require more than one bin
    exams["age"] = exams["age_days"].transform(lambda days: timedelta(days=days))

    observations = ObservationMatrix().fit_transform(exams)
    assert observations.dtypes["bin"] == "int"

    age_min = exams["age"].min()
    age_max = exams["age"].max()
    assert observations["bin"].max() == ceil((age_max - age_min).days / 30 / 3) - 1


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
    }
    assert set(person_df.columns) == expected_columns

    invalid_pids = [
        7,
        15,
        21,
        27,
        31,
        42,
        49,
        50,
        56,
        57,
        59,
        60,
        62,
        68,
        72,
        79,
        80,
        83,
        86,
        93,
        94,  # Not present in screening data
        61,  # Invalid DOB data
    ]  # People not present in data or with invalid data
    expected_pids = {i for i in range(1, 101) if i not in invalid_pids}
    assert expected_pids == set(person_df.index)


def test_person_stats_w_features():
    raw = read_raw_df(test_data_screening)

    base_pipeline = get_base_pipeline(
        birthday_file=test_data_dob, drop_missing_birthday=True
    )
    pipeline = get_exam_pipeline(base_pipeline=base_pipeline)
    base_df = base_pipeline.fit_transform(raw)
    exams: pd.DataFrame = pipeline.fit_transform(raw)

    person_df = PersonStats(base_df=base_df).fit_transform(exams)
    logger.debug(f"Person df:\n {person_df}")
    logger.debug(f"Person df columns: {list(person_df.columns)}")

    expected_columns = {
        # Base columns
        "age_min",
        "age_max",
        "age_mean",
        "risk_min",
        "risk_max",
        "risk_mean",
        "FOEDT",
        # Feature columns
        "count_positive",
        "count_negative",
        "number_of_screenings",
        "age_last_exam",
        "age_first_positive",
        "age_first_negative",
        "count_positive_last_5_years",
        "count_negative_last_5_years",
    }
    assert set(person_df.columns) == expected_columns

    int_features = [
        "count_positive",
        "count_negative",
        "number_of_screenings",
        "count_positive_last_5_years",
        "count_negative_last_5_years",
    ]
    float_features = [
        "age_last_exam",
        "age_first_positive",
        "age_first_negative",
    ]
    for feature in int_features:
        assert person_df[feature].dtype == "int"
    for feature in float_features:
        assert person_df[feature].dtype == "float"

    has_positive = person_df["count_positive"] != 0
    assert person_df.loc[has_positive, "age_first_positive"].min() > 0
    assert not person_df.loc[has_positive, "age_first_positive"].isna().any()

    has_negative = person_df["count_negative"] != 0
    assert person_df.loc[has_negative, "age_first_negative"].min() > 0
    assert not person_df.loc[has_negative, "age_first_negative"].isna().any()

    PID_to_last_exam = person_df["age_last_exam"].apply(
        lambda x: timedelta(days=x * 365)
    )
    possible_wrong_exams = exams["age"] > exams["PID"].map(PID_to_last_exam)
    assert (exams[possible_wrong_exams]["exam_type"] == "HPV").all()


def test_hpv_results():
    raw = read_raw_df(test_data_screening)
    hpv_df = HPVResults().fit_transform(raw)
    logger.debug(f"HPV DF:\n{hpv_df.head()}")

    assert not hpv_df.isna().any().any()

    genotype_columns = ["hpv1Genotype", "hpv2Genotype"]

    # Check that the data corresponds with the raw data
    for exam_index, results in hpv_df.groupby("exam_index"):
        raw_row = raw.loc[exam_index]

        assert set(results["genotype"]) == set(raw_row[genotype_columns].dropna())

        def _matches(field: str) -> bool:
            """Assert the field is unique within the group and matches the raw data"""
            return (
                results[field].nunique() == 1
                and results[field].iloc[0] == raw_row[field]
            )

        for field in ("PID", "hpvTesttype", "hpvDate"):
            assert _matches(field)

    # Check that the test type names are correct
    assert hpv_df["test_type_name"].equals(
        hpv_df["hpvTesttype"].map(HPV_TEST_TYPE_NAMES).astype("category")
    )


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


pid_strategy = st.integers(min_value=0, max_value=100)
hpv_type_strategy = st.integers(min_value=1, max_value=100)


def date_time_strategy():
    return st.datetimes(min_value=datetime(2010, 1, 1), max_value=datetime(2023, 1, 1))


@given(
    last_dates=st.dictionaries(
        keys=pid_strategy, values=date_time_strategy(), min_size=1
    ),
    data_strategy=st.data(),
    time_window=st.timedeltas(
        min_value=timedelta(days=1), max_value=timedelta(days=5 * 365)
    ),
)
def test_count_in_time_window(last_dates, data_strategy, time_window):
    unique_pids = sorted(last_dates)  # Sorted to make sure the test is reproducible

    sample_pids = data_strategy.draw(st.lists(st.sampled_from(unique_pids), min_size=1))
    sample_times = data_strategy.draw(
        st.lists(
            date_time_strategy(), min_size=len(sample_pids), max_size=len(sample_pids)
        ),
    )

    result = PersonStats.count_in_time_window(
        times=pd.DataFrame({"PID": sample_pids, "time": sample_times}),
        last_date=pd.Series(last_dates, dtype="datetime64[ns]"),
        time_window=time_window,
    )

    for pid, count in result.items():
        assert count == len(
            [
                id
                for id, date in zip(sample_pids, sample_times)
                if id == pid
                and last_dates[pid] >= date >= last_dates[pid] - time_window
            ]
        )
