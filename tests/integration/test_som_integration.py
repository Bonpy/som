from pathlib import Path

import numpy as np
import pytest

from colour_som import SOMFast

this_dir = Path(__file__).absolute().parent

GRID_SIZE = (10, 10)
INPUT_DIM = 3

TEST_PARAMS = [
    {"lr": 0.1, "n_iterations": 50, "data_seed": 123},
    {"lr": 0.01, "n_iterations": 100, "data_seed": 456},
    {"lr": 0.5, "n_iterations": 20, "data_seed": 789},
]


@pytest.mark.parametrize("params", TEST_PARAMS)
def test_som_integration(params):
    """
    Integration test for the complete SOM workflow with different learning rates, iteration counts,
    and datasets.

    Parameters:
        params (dict): Dictionary containing learning rate (lr),
        number of iterations (n_iterations), and data seed (data_seed).

    This test runs the entire SOM process: initialization, training, and evaluating the
    quantization error, comparing it to expected values.
    """

    # setup test data and SOM
    np.random.seed(params["data_seed"])
    test_data = np.random.rand(100, INPUT_DIM)
    som = SOMFast(GRID_SIZE, INPUT_DIM)

    # training
    som.train(test_data, n_iterations=params["n_iterations"], lr=params["lr"])

    # compar quantization error with expected value
    quant_error = som.quantization_error(test_data)
    expected_error_path = this_dir.joinpath(
        f"expected_outputs/expected_quant_error_{params['lr']}_{params['n_iterations']}_{params['data_seed']}.npy"
    )
    expected_error = np.load(expected_error_path)
    assert abs(quant_error - expected_error) < 0.01
