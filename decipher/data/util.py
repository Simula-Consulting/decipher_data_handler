from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def _last_observed_time(observation_data: npt.NDArray) -> npt.NDArray:
    """Return the last observed entry per row."""
    max_time = observation_data.shape[1]
    return max_time - np.argmax(observation_data[:, ::-1] != 0, axis=1) - 1


def _mask_row(row: npt.NDArray, time_point: int, mask_window_size: int = 3) -> None:
    """Mask the `mask_window_size` points up until `time_point` in-place.

    Warning:
        The row is masked in-place!
    """
    row[max(0, time_point - mask_window_size) :] = 0


def prediction_data(
    observation_matrix: npt.NDArray,
    prediction_time_strategy: Callable[
        [npt.NDArray], npt.NDArray
    ] = _last_observed_time,
    row_masking_strategy: Callable[[npt.NDArray, int], None] = _mask_row,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Mask observed data to produce test data for predictions.

    Arguments:
        prediction_time_strategy: Callable returning prediction times given the
            observation_matrix. Defaults to choosing the last observation.
        row_masking_strategy: Callable masking an observation row,
            i.e. individual history, given a time point. Note that the masking is done
            in place, not by return.
            Defaults to masking the three time slots from prediction time and back.

    Examples:
        >>> masked_observation, times, correct_values = prediction_data(my_data)
        >>> my_data[range(times.size), times] == correct_values
        True
        >>> (masked_observation[range(times.size), times] == 0).all()
        True
    """

    times = prediction_time_strategy(observation_matrix)
    # Choose the value at times for each row
    correct_values = np.take_along_axis(
        observation_matrix, times[:, None], axis=1
    ).flatten()

    # row_masking_strategy masks in-place, so important to copy!
    masked_observation_matrix = np.copy(observation_matrix)
    for i, row in enumerate(masked_observation_matrix):
        row_masking_strategy(row, times[i])

    return masked_observation_matrix, times, correct_values
