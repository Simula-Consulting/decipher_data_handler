import logging
from pathlib import Path

import pytest

from decipher.data import DataManager

logger = logging.getLogger(__name__)

test_data_base = Path(__file__).parent / "test_processing_datasets"
test_data_screening = test_data_base / "test_screening_data.csv"
test_data_dob = test_data_base / "test_dob_data.csv"


@pytest.fixture()
def data_manager() -> DataManager:
    return DataManager(test_data_screening, test_data_dob)


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
