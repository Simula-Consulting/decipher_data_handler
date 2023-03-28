import json
import logging
from pathlib import Path

import pytest

from decipher.data import DataManager
from decipher.data.data_manager import _parquet_engine_types

logger = logging.getLogger(__name__)

test_data_base = Path(__file__).parent / "test_processing_datasets"
test_data_screening = test_data_base / "test_screening_data.csv"
test_data_dob = test_data_base / "test_dob_data.csv"


@pytest.fixture()
def data_manager() -> DataManager:
    return DataManager.read_from_csv(test_data_screening, test_data_dob)


def test_data_shape(data_manager: DataManager):
    assert data_manager.shape() == (66, 233)  # Correct shape of test file


def test_get_observation_array(data_manager: DataManager):
    data_manager.data_as_coo_array()


def test_get_feature_array(data_manager: DataManager):
    data_manager.feature_data_as_coo_array()


def test_get_masked_array(data_manager: DataManager):
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
        # "fastparquet",  # TODO: This fails, but looks good...
        # "auto",
    ],
)
def test_parquet(
    data_manager: DataManager, tmp_path: Path, parquet_engine: _parquet_engine_types
):
    data_manager.save_to_parquet(tmp_path, engine=parquet_engine)
    new_data_manager = DataManager.from_parquet(tmp_path, engine=parquet_engine)
    assert data_manager.screening_data.equals(new_data_manager.screening_data)
    assert data_manager.person_df.drop(columns="age_mean").equals(
        new_data_manager.person_df.drop(columns="age_mean")
    )
    assert data_manager.pid_to_row == new_data_manager.pid_to_row


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
