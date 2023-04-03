import json
import logging
import operator
from datetime import timedelta
from pathlib import Path

import pytest

from decipher.data import DataManager
from decipher.data.data_manager import (
    AtLeastNNonHPV,
    CombinePersonFilter,
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


@pytest.mark.parametrize("min_screenings_to_person", [(0, 78), (2, 66), (3, 47)])
def test_data_shape(
    data_manager: DataManager, min_screenings_to_person: tuple[int, int]
):
    min_screenings, expected_number_of_people = min_screenings_to_person
    filter_ = AtLeastNNonHPV(min_n=min_screenings)
    data_manager.get_screening_data(filter_strategy=filter_, update_inplace=True)
    assert data_manager.shape() == (
        expected_number_of_people,
        233,
    )  # Correct shape of test file


def test_get_observation_array(data_manager: DataManager):
    with pytest.raises(ValueError):  # Screening data not implemented yet
        data_manager.data_as_coo_array()
    with pytest.raises(ValueError):  # Screening data not implemented yet
        data_manager.get_screening_data()
        data_manager.data_as_coo_array()
    data_manager.get_screening_data(update_inplace=True)
    data_manager.data_as_coo_array()


def test_get_feature_array(data_manager: DataManager):
    with pytest.raises(ValueError):  # Screening data not implemented yet
        data_manager.feature_data_as_coo_array()
    data_manager.get_screening_data(update_inplace=True)
    feature_array = data_manager.feature_data_as_coo_array()
    min_non_hpv_exams = 2
    number_of_people = len(
        data_manager.person_df[
            data_manager.person_df[["cytology_count", "histology_count"]].sum(axis=1)
            >= min_non_hpv_exams
        ].index
    )
    number_of_features = 4
    assert feature_array.shape == (number_of_people, number_of_features)


def test_get_masked_array(data_manager: DataManager):
    with pytest.raises(ValueError):  # Screening data not implemented yet
        data_manager.get_masked_data()
    data_manager.get_screening_data(update_inplace=True)

    masked_X, t_pred, y_true = data_manager.get_masked_data()
    X = data_manager.data_as_coo_array()

    X = X.toarray()
    masked_X = masked_X.toarray()
    for i, (t, y) in enumerate(zip(t_pred, y_true)):
        assert X[i, t] == y
        assert masked_X[i, t] == 0


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
        assert data_manager.pid_to_row is not None
        assert data_manager.screening_data.equals(
            new_data_manager.screening_data  # type: ignore[arg-type]
        )
        assert data_manager.pid_to_row == new_data_manager.pid_to_row
    else:
        assert new_data_manager.screening_data is None
        assert new_data_manager.pid_to_row is None
    assert data_manager.person_df.equals(new_data_manager.person_df)
    assert data_manager.exams_df.equals(new_data_manager.exams_df)


@pytest.mark.parametrize("min_number_of_screenings", [0, 1, 3])
def test_metadata(data_manager: DataManager, min_number_of_screenings: int):
    data_manager.get_screening_data(
        filter_strategy=AtLeastNNonHPV(min_n=min_number_of_screenings),
        update_inplace=True,
    )
    assert (
        data_manager.metadata["screenings_filters"]["min_n"] == min_number_of_screenings
    )


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
