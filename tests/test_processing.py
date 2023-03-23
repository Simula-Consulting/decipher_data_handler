from pathlib import Path

import pytest

from decipher.processing.pipeline import exam_pipeline, read_raw_df

test_data_base = Path(__file__).parent / "test_processing_datasets"
test_data_screening = test_data_base / "test_screening_data.csv"
test_data_dob = test_data_base / "test_dob_data.csv"


def test_read_and_pipeline():
    """Simply try reading and running pipeline"""
    raw = read_raw_df(test_data_screening)

    # This should fail as we have people with missing birth data
    with pytest.raises(ValueError):
        pipeline = exam_pipeline(
            birthday_file=test_data_dob, drop_missing_birthday=False
        )
        pipeline.fit_transform(raw)

    pipeline = exam_pipeline(birthday_file=test_data_dob, drop_missing_birthday=True)
    pipeline.fit_transform(raw)
