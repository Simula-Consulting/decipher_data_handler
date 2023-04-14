"""Script for easily testing out baselines."""
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import rich
import seaborn as sns
import typer
from loguru import logger
from sklearn.metrics import f1_score, matthews_corrcoef

from decipher.data import DataManager


# Redirect logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

app = typer.Typer()


def compute_score(y_true, prediction) -> dict[str, float | int]:
    """Compute the score given a truth and prediction."""
    prediction_int = np.around(prediction).clip(1, 4)
    return {
        "norm": np.linalg.norm(prediction - y_true) / len(y_true),
        "f1_micro": f1_score(y_true, prediction_int, average="micro"),
        "f1_weighted": f1_score(y_true, prediction_int, average="weighted"),
        "MCC": matthews_corrcoef(y_true, prediction_int),
    } | {
        f"f1_per_class__{i}": score
        for i, score in enumerate(f1_score(y_true, prediction_int, average=None))
    }


@app.command()
@logger.catch
def cache_data(
    screening_data_path: Path,
    dob_data_path: Path,
    parquet_path: Path,
    load_screening_data: bool = True,
):
    """Read data from the CSVs, and cache in a DataManager friendly parquet_format."""

    logger.info(f"Screening data path: {screening_data_path}")
    logger.info(f"DOB data path: {dob_data_path}")
    logger.info(f"Parquet path: {parquet_path}")

    assert screening_data_path.exists() and dob_data_path.exists()
    assert parquet_path.is_dir() and not list(parquet_path.iterdir())

    data_manager = DataManager.read_from_csv(screening_data_path, dob_data_path)
    logger.info("Read the data manager")
    if load_screening_data:
        data_manager.get_screening_data(update_inplace=True)
        logger.info("Got the screening data")
    logger.debug(f"The DataManager's metadata is: {data_manager.metadata}")
    data_manager.save_to_parquet(parquet_path)
    logger.info("Saved the data manager")


@app.command()
@logger.catch
def run_baseline(parquet_path: Path, default_guess: float = 1.0):
    """Predict using the baseline.

    Current implementation simply sets all predictions to the same number.

    Arguments:
    - parquet_path: path of the cached data manager
    - default_guess: the prediction to use for the baseline
    """
    logger.info("Running baseline")
    logger.debug("Getting ready to load DataManager")
    data_manager = DataManager.from_parquet(parquet_path)
    logger.info("Loaded DataManager")
    logger.info(f"DataManager metadata:\n{data_manager.metadata}")
    logger.debug("Getting ready to load data")
    (
        _,
        _,
        y_true,
    ) = data_manager.get_masked_data()  # We don't care about the screening data
    logger.info("Loaded data")

    scores = compute_score(y_true, np.full_like(y_true, default_guess))
    rich.print(scores)

    rich.print("True states: ", Counter(y_true))
    return scores


def _count_number_screenings(data_manager: DataManager) -> list[int]:
    """Count number of screenings per person"""
    X = data_manager.data_as_coo_array().toarray()
    return np.count_nonzero(X, axis=1)


@app.command()
@logger.catch
def sanity_check(parquet_path: Path):
    """Run various sanity checks on the DataManager's data"""

    logger.debug("Getting ready to load DataManager")
    data_manager = DataManager.from_parquet(parquet_path)
    logger.info("Loaded DataManager")
    logger.info(f"DataManager metadata:\n{data_manager.metadata}")
    logger.debug("Getting ready to load data")
    X_masked, t_pred, y_true = data_manager.get_masked_data()
    logger.info("Loaded data")
    logger.debug(f"Number of people: {len(y_true)}")

    issues = []

    # At least two screening results
    number_of_screenings = _count_number_screenings(data_manager)
    if min(number_of_screenings) < 2:
        logger.critical("There are people with less than 2 screenings.")
        counts = Counter(number_of_screenings)
        logger.debug(f"The counts are: {counts}")
        issues.append(
            {
                "title": "There are people with less than 2 screenings",
                "data": {"counts": counts},
            }
        )

    # Only 1-4 states in prediction set
    states_to_be_predicted = set(np.unique(y_true))
    if states_to_be_predicted > set(range(1, 5)):
        logger.critical(
            f"There are states to predicted not in [1-4]! They are {states_to_be_predicted}."
        )
        logger.debug(f"Number of 0s: {np.count_nonzero(y_true == 0)}")
        issues.append(
            {
                "title": "There are states to predicted not in [1-4]!",
                "data": states_to_be_predicted,
            }
        )

    # Only 0-4 states in total set
    observed_states = set(np.unique(X_masked.data))
    if observed_states > set(range(5)):
        logger.critical(
            f"There are other states in the observation matrix than [0-4]! They are {observed_states}"
        )
        issues.append(
            {
                "title": "There are other states in the observation matrix than [0-4]!",
                "data": observed_states,
            }
        )

    # No duplicate row/col
    row_col_list = list(zip(X_masked.row, X_masked.col))
    if len(row_col_list) != len(set(row_col_list)):
        logger.critical("There are duplicate entries in the observation matrix!!")
        issues.append(
            {"title": "There are duplicate entries in the observation matrix!!"}
        )

    rich.print(issues)


@app.command()
@logger.catch
def boxplot(
    experiment_ids: list[str],
    mlflow_tracking_uri: str = f"sqlite:////{Path.home()}/runs.db",
    filter_string: str = "",
    metrics: Optional[list[str]] = typer.Option(None, "--metric", "-m"),
    use_hue: bool = True,
    plot_swarm: bool = True,
):
    metrics = metrics or ["norm", "f1_micro", "f1_weighted"]
    logger.debug(f"Logging with metrics {metrics}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.debug(f"Set tracking uri to {mlflow_tracking_uri}")
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
    )
    if not len(runs.index):
        logger.critical("No runs found!")
        raise ValueError("No runs")
    logger.info(f"Found {len(runs.index)} runs")
    logger.debug(f"Runs:\n{runs.head()}")

    hue_field = "params.cmfrec__center"

    def _plot_and_save(df, hue, name, output_extension, include_swarm_plot=True):
        boxplot = sns.boxplot(
            data=df, x="params.use_hpv_data", y=f"metrics.{metric}", hue=hue
        )
        if include_swarm_plot:
            sns.swarmplot(
                data=df,
                x="params.use_hpv_data",
                y=f"metrics.{metric}",
                hue=hue,
                color="black",
                marker="d",
                alpha=0.5,
                size=8,
                dodge=True,
            )
        boxplot.set(title=name)
        boxplot.get_figure().savefig(f"{name}.{output_extension}")
        boxplot.get_figure().clf()

    for metric in metrics:
        if not use_hue:
            subdata = dict(iter(runs.groupby(hue_field)))
            for hue_value, df in subdata.items():
                _plot_and_save(
                    df,
                    hue=None,
                    name=f"boxplot_{metric}__{hue_field}_{hue_value}",
                    output_extension="png",
                    include_swarm_plot=plot_swarm,
                )
                logger.info(f"Plotted boxplot for {metric} and {hue_field}={hue_value}")
        else:
            _plot_and_save(
                runs,
                hue=hue_field,
                name=f"boxplot_{metric}",
                output_extension="png",
                include_swarm_plot=plot_swarm,
            )
            logger.info(f"Plotted boxplot for {metric}")
