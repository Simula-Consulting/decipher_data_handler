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

#### The feature matrix

The feature matrix is extracted as
```python
data_manager.feature_data_as_coo_array()
```
or select only a subset of the features by passing a list of features to choose from.
These features have to be columns of `data_manager.person_df`.
By default, these are `["has_positive", "has_negative", "has_hr", "has_hr_2"]`.
To use other features, or to alter these, add columns as needed in the person data frame.

For example
```python
from pathlib import Path
from decipher.data import DataManager

parquet_dir = Path(<parquet dir>)

# Read from Parquet
data_manager = DataManager.from_parquet(parquet_dir, engine="pyarrow")

# Add feature
data_manager.person_df["risky"] = data_manager.person_df["risk_max"] > 2

# Get out the feature matrix
# We include also the non-default 'risky' feature and drop 'has_hr2'.
feature_matrix = data_manager.feature_data_as_coo_array(
    cols=["has_positive", "has_negative", "has_hr", "risky"]
)
```

### Recipes

** Adding detailed HPV test type and result information to the `exams_df` **
The `exams_df` of the `DataManger` only contains whether the HPV result was positive
or not, and no specific information about the test type.
This information is stored in `hpv_df` (which is only populated if `read_hpv` is set
to `True` in `DataManager.read_from_csv`).

In some cases, it is desirable to have this information in the `exams_df`, as new columns.
We here do it in two 'steps' to more clearly show what is going on.

```python
def hpv_details_per_exam(hpv_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the HPV details per exam.

    The index of the returned DataFrame is the exam_index. Note that this is
    not the same as the index of the exams_df!"""

    per_exam = hpv_df.groupby("exam_index")
    if per_exam["hpvTesttype"].nunique().max() != 1:
        raise ValueError("Not all exams have the same HPV test type!")

    return pd.DataFrame(
        {
            "exam_detailed_type": per_exam["test_type_name"].first(),
            "exam_detailed_results": per_exam["genotype"].apply(
                lambda genotypes: ",".join(genotypes)
            ),
        }
    )

def add_hpv_detailed_information(
    exams_df: pd.DataFrame, hpv_df: pd.DataFrame
) -> pd.DataFrame:
    """ "Add detailed exam type name and results to exams_df"""
    # Find the exam_index -> exams_df.index map
    # exam_index is not unique in exams_df, because one exam may give
    # cyt, hist, and HPV results
    # Therefore, we find the indices where there is an HPV test
    hpv_indices = exams_df.query("exam_type == 'HPV'")["index"]
    mapping = pd.Series(data=hpv_indices.index, index=hpv_indices.values)

    hpv_details = hpv_details_per_exam(hpv_df)
    hpv_details.index = hpv_details.index.map(mapping)

    # TODO: this will give nan on the hist and cyt rows
    exams_df = exams_df.join(hpv_details)

    # Set the Cytology and Histology results
    def _fill(base_series: pd.Series, fill_series: pd.Series) -> pd.Series:
        """Fill base series with fill series where base series is nan. Handles category data."""
        return base_series.astype("string").fillna(fill_series.astype("string"))

    exams_df["exam_detailed_type"] = _fill(
        exams_df["exam_detailed_type"], exams_df["exam_type"]
    )
    exams_df["exam_detailed_results"] = _fill(
        exams_df["exam_detailed_results"], exams_df["exam_diagnosis"]
    )

    return exams_df


# Assuming the DataManger has hpv_df
data_manager.exams_df = add_hpv_detailed_information(data_manager.exams_df, data_manager.hpv_df)
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
