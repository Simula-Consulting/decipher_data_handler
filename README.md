# Decipher util package

## Usage

### Read the raw screening data

```python
from pathlib import Path
from decipher.processing.pipeline import read_raw_df

screening_data = Path(<screening_data_path>)
raw = read_raw_df(screening_data)
```

### Create exams df

```python
from pathlib import Path
from decipher.processing.pipeline import read_raw_df, exam_pipeline

screening_data = Path(<screening_data_path>)
dob_data = Path(<date of birth path>)
raw = read_raw_df(screening_data)

pipeline = exam_pipeline(birthday_file=dob_data)
exams = pipeline.fit_transform(raw)
```

### Get person table with stats

```python
from pathlib import Path
from decipher.processing.pipeline import read_raw_df, exam_pipeline
from decipher.processing.transformers import PersonStats

screening_data = Path(<screening_data_path>)
dob_data = Path(<date of birth path>)
raw = read_raw_df(screening_data)

pipeline = exam_pipeline(birthday_file=dob_data)
exams = pipeline.fit_transform(raw)

person_df = PersonStats().fit_transform(exams)
```

### Get HPV results

```python
from pathlib import Path
from decipher.processing.pipeline import read_raw_df
from decipher.processing.transformers import HPVResults

screening_data = Path(<screening_data_path>)
raw = read_raw_df(screening_data)

hpv_df = HPVResults().fit_transform(raw)
```

### Writing and reading DataFrames with metadata

```python
from pathlib import Path
from decipher.processing.pipeline import read_from_csv, write_to_csv

data_path = Path(<data_path>)

df, metadata = read_from_csv(data_path)
df = do_something(df)
metadata['notes'].append("Fixed xxx")

write_to_csv(data_path, df=df, metadata=metadata)
```
Note that the `decipher` version will always be written to the metadata.

## Install

## Parquet support
For loading and dumping to the [Parquet format](https://parquet.apache.org/), `pyarrow` or `fastparquet` is needed.
They may be installed by
```bash
pip install decipher.whl[pyarrow]
# or
pip install decipher.whl[fastparquet]
```
if using pip from the wheel.
From Poetry, do
```bash
poetry install -E <pyarrow|fastparquet>
```


Note that, at the moment, `pyarrow` is the only we guarantee support for.

## Other tips and notes

### Debugging info in tests

Add
```toml
[tool.pytest.ini_options]
log_cli = true
log_cli_level = "DEBUG"
```
to the `pyproject.toml` to get debugging details when running the tests.
