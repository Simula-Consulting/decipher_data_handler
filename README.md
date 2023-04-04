# Decipher util package

## Usage

### `DataManager`
While there are many lower level tools, `DataManger` is the simplest interface to use.

**Read from CSVs**

```python
from pathlib import Path
from decipher.data import DataManager

screening_data = Path(<screening data>)
dob_data = Path(<dob data>)

# Read in from CSV
data_manager = DataManager.read_from_csv(screening_data, dob_data)
```

**Read and write with Parquet**

```python
from pathlib import Path
from decipher.data import DataManager

screening_data = Path(<screening data>)
dob_data = Path(<dob data>)
parquet_dir = Path(<parquet dir>)

# Read in from CSV
data_manager = DataManager.read_from_csv(screening_data, dob_data)

# Store to Parquet
data_manager.save_to_parquet(parquet_dir, engine="pyarrow")

# Read from Parquet
# Will fail if `decipher` version does not match that of stored data
data_manager = DataManager.from_parquet(parquet_dir, engine="pyarrow")

# See metadata
data_manager.metadata
```

**Create the observation matrix**

```python
from pathlib import Path
from decipher.data import DataManager

parquet_dir = Path(<parquet dir>)

# Read from Parquet
data_manager = DataManager.from_parquet(parquet_dir, engine="pyarrow")

# update_inplace updates the data manager, instead of only
# returning the values
data_manager.get_screening_data(min_non_hpv_exams=3, update_inplace=True)

# Saving will now also include the computed observation data
# Not required, just convenient
data_manger.save_to_parquet(parquet_dir, engine="pyarrow")

# Actually get the matrix
X = data_manager.data_as_coo_array()
# or
X_masked, t_pred, y_true = data_manager.get_masked_data()
```

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
