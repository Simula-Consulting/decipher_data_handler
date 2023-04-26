import json
import logging
import operator
from datetime import timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.sparse import coo_array

from decipher.data import DataManager
from decipher.data.data_manager import (
    AtLeastNNonHPV,
    CombinePersonFilter,
    MaximumTimeSeparation,
    OperatorFilter,
    PersonFilter,
    TrueFields,
    _parquet_engine_types,
)

logger = logging.getLogger(__name__)

test_data_base = Path(__file__).parent / "test_processing_datasets"
test_data_screening = test_data_base / "test_screening_data.csv"
test_data_dob = test_data_base / "test_dob_data.csv"


@pytest.fixture()
def data_manager() -> DataManager:
    return DataManager.read_from_csv(test_data_screening, test_data_dob)


@pytest.mark.parametrize("min_non_hpv_exams", [0, 2, 3])
def test_get_feature_data(data_manager: DataManager, min_non_hpv_exams: int):
    filter_ = AtLeastNNonHPV(
        min_n=min_non_hpv_exams
    )  # Some other value than default, to uncover assumptions
    data_manager.get_screening_data(filter_strategy=filter_, update_inplace=True)

    pids = np.random.choice(
        data_manager.person_df.index,
        int(0.7 * len(data_manager.person_df.index)),
        replace=False,
    )
    columns = ["has_positive", "has_hr"]
    features = data_manager.get_feature_data(pids=pids, columns=columns)

    assert set(features["PID"]) == set(pids)
    assert set(features["feature"]) == set(columns)
    assert not features["value"].isna().any()

    # Test that the recipe given in the example of the docstring works
    pid_to_row = {pid: i for i, pid in enumerate(pids)}
    feature_to_col = {feature: i for i, feature in enumerate(columns)}
    feature_array = coo_array(
        (
            features["value"],
            (features["PID"].map(pid_to_row), features["feature"].map(feature_to_col)),
        ),
        shape=(len(pid_to_row), len(feature_to_col)),
        dtype="int8",
    )

    assert feature_array.nnz == len(features)


def test_get_last_screening_bin(data_manager: DataManager):
    data_manager.get_screening_data(update_inplace=True)
    last_screening_bin = data_manager.get_last_screening_bin()
    assert data_manager.screening_data is not None

    for pid in data_manager.screening_data["PID"].unique():
        assert (
            last_screening_bin[pid]
            == data_manager.screening_data[data_manager.screening_data["PID"] == pid][
                "bin"
            ].max()
        )


def test_get_screening_data(data_manager: DataManager):
    data_manager.get_screening_data(update_inplace=True)
    assert data_manager.screening_data is not None

    assert data_manager.screening_data is not None
    assert not data_manager.screening_data["risk"].isna().any()
    assert data_manager.screening_data["risk"].isin(range(1, 5)).all()

    last_screening_bin = data_manager.get_last_screening_bin()
    for pid, group in data_manager.screening_data.groupby("PID"):
        pid = cast(int, pid)
        assert group["is_last"].sum() == 1
        assert group[group["is_last"]]["bin"].item() == last_screening_bin[pid]


@pytest.mark.parametrize(
    "parquet_engine",
    [
        "pyarrow",
        # "fastparquet",  # TODO: This fails, but looks good... Some round off error?
        # "auto",
    ],
)
@pytest.mark.parametrize("initialize_screening_data", [True, False])
def test_parquet(
    data_manager: DataManager,
    tmp_path: Path,
    parquet_engine: _parquet_engine_types,
    initialize_screening_data: bool,
):
    if initialize_screening_data:
        data_manager.get_screening_data(update_inplace=True)
    data_manager.save_to_parquet(tmp_path, engine=parquet_engine)
    new_data_manager = DataManager.from_parquet(tmp_path, engine=parquet_engine)
    if initialize_screening_data:
        assert data_manager.screening_data is not None
        assert data_manager.screening_data.equals(
            new_data_manager.screening_data  # type: ignore[arg-type]
        )
    else:
        assert new_data_manager.screening_data is None
    assert data_manager.person_df.equals(new_data_manager.person_df)
    assert data_manager.exams_df.equals(new_data_manager.exams_df)


@pytest.mark.parametrize("min_number_of_screenings", [0, 1, 3])
def test_metadata(data_manager: DataManager, min_number_of_screenings: int, tmp_path):
    data_manager.get_screening_data(
        filter_strategy=AtLeastNNonHPV(min_n=min_number_of_screenings),
        update_inplace=True,
    )
    assert (
        data_manager.metadata["screenings_filters"]["min_n"] == min_number_of_screenings
    )
    data_manager.save_to_parquet(tmp_path)
    new_data_manger = DataManager.from_parquet(tmp_path)
    assert data_manager.metadata == new_data_manger.metadata


@pytest.mark.parametrize(
    "filter_",
    [
        OperatorFilter("age_min", operator=operator.lt, value=timedelta(days=365 * 50)),
        TrueFields(["has_hr"]),
        TrueFields(["has_hr", "has_positive"]),
        CombinePersonFilter(
            [
                OperatorFilter(
                    "age_min", operator=operator.lt, value=timedelta(days=365 * 50)
                ),
                TrueFields(["has_hr"]),
            ]
        ),
    ],
)
def test_filter(data_manager: DataManager, filter_: PersonFilter):
    data_manager.get_screening_data(filter_strategy=filter_, update_inplace=True)
    logger.debug(f"Metadata is:\n{data_manager.metadata}")


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(
    filters=st.lists(
        st.sampled_from(
            [
                OperatorFilter(
                    "age_min", operator=operator.lt, value=timedelta(days=365 * 50)
                ),
                OperatorFilter("risk_min", operator=operator.gt, value=1),
                TrueFields(["has_hr"]),
                TrueFields(["has_hr", "has_positive"]),
                AtLeastNNonHPV(2),
                AtLeastNNonHPV(4),
            ]
        ),
        max_size=4,
        min_size=1,
    )
)
def test_combine_filter(data_manager: DataManager, filters: list[PersonFilter]):
    combined_filter = CombinePersonFilter(filters)
    screening_data, _ = data_manager.get_screening_data(filter_strategy=combined_filter)
    pids = set(screening_data["PID"])
    for filter in filters:
        other_filter_screening_data, _ = data_manager.get_screening_data(
            filter_strategy=filter
        )
        assert pids <= set(other_filter_screening_data["PID"])


@pytest.mark.parametrize("min_n_exams", [0, 2, 4])
def test_filter_2(data_manager: DataManager, min_n_exams: int):
    data_manager.get_screening_data(
        filter_strategy=AtLeastNNonHPV(min_n=min_n_exams), update_inplace=True
    )
    assert data_manager.screening_data is not None

    counts_per_person = data_manager.screening_data["PID"].value_counts()
    assert (counts_per_person >= min_n_exams).all()


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(
    max_time_difference=st.timedeltas(
        min_value=timedelta(days=0), max_value=timedelta(days=365 * 7)
    )
)
def test_maximum_time_separation_filter(
    data_manager: DataManager, max_time_difference: timedelta
):
    # max_time_difference = timedelta(days=365 * 2)
    filter_ = MaximumTimeSeparation(max_time_difference=max_time_difference)

    pids = filter_.filter(data_manager.person_df, data_manager.exams_df)

    filtered_exams = data_manager.exams_df[data_manager.exams_df["PID"].isin(pids)]
    assert (filtered_exams["PID"].value_counts() >= 2).all()

    differences = (
        filtered_exams.groupby("PID")["age"]
        .nlargest(2)
        .groupby("PID")
        .agg(lambda x: max(x) - min(x))
    )
    assert (differences <= max_time_difference).all()


def test_parquet_version(data_manager: DataManager, tmp_path: Path):
    data_manager.save_to_parquet(tmp_path)
    with open(tmp_path / "metadata.json", "r+") as file:
        metadata = json.load(file)
        metadata["decipher_version"] = "foo"
        file.seek(0)
        json.dump(metadata, file)
        file.truncate()
    with pytest.raises(ValueError):
        DataManager.from_parquet(tmp_path)
    DataManager.from_parquet(tmp_path, ignore_decipher_version=True)
