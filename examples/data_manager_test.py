import logging
from pathlib import Path

from decipher.data import DataManager

SCREENING_DATA_PATH = Path("screening_data.csv")
DOB_DATA_PATH = Path("dob_data.csv")

logging.getLogger("decipher.data.data_manager").setLevel(logging.DEBUG)


def main():
    DataManager.read_from_csv(
        screening_path=SCREENING_DATA_PATH, dob_path=DOB_DATA_PATH
    )


if __name__ == "__main__":
    main()
