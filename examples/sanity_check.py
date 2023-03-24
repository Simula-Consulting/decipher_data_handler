"""Example to be run on TSD as a sanity check of both the data and the package."""

import logging
from pathlib import Path

import pandas as pd

from decipher.processing.pipeline import get_exam_pipeline, read_raw_df
from decipher.processing.transformers import HPVResults, PersonStats

logger = logging.getLogger(__name__)

data_base = Path("/mnt/durable/003-jfn/Decipher/data/NewData_2022")
screening_data = data_base / "lp_pid_fuzzy.csv"
dob_data = data_base / "Folkereg_PID_fuzzy.csv"

assert screening_data.exists() and dob_data.exists()

raw = read_raw_df(screening_data)

pipeline = get_exam_pipeline(birthday_file=dob_data)

exams: pd.DataFrame = pipeline.fit_transform(raw)
person_df: pd.DataFrame = PersonStats().fit_transform(exams)
hpv_df: pd.DataFrame = HPVResults().fit_transform(raw)

## Check hpv sanity
hpv_people = set(person_df.query("HPV_count >= 1").index)
if hpv_people != set(hpv_df["PID"].unique()):
    logger.critical("People in person_df with hpv not equal to hpv_df")
