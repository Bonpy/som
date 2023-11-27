from pathlib import Path

import numpy as np
import pytest

from colour_som import SOMFast

this_dir = Path(__file__).absolute().parent

GRID_SIZE = (10, 10)
INPUT_DIM = 3
np.random.seed(123)
TEST_DATA = np.random.rand(100, INPUT_DIM)


@pytest.mark.parametrize("lr,n_iterations", [(0.1, 10), (0.01, 100), (0.5, 20)])
def test_train_regression_different_params(lr, n_iterations):
    """
    Regression test for training the SOM with different learning rates and iteration counts.

    Parameters:
        lr (float): Learning rate to test.
        n_iterations (int): Number of iterations to test.

    This test compares the SOM's trained weights with expected weights for different combinations
    of learning rates and iteration counts.
    """
    som = SOMFast(GRID_SIZE, INPUT_DIM)
    som.train(TEST_DATA, n_iterations=n_iterations, lr=lr)
    expected_weights_path = this_dir.joinpath(
        f"expected_outputs/expected_weights_diff_params_{lr}_{n_iterations}.npy"
    )
    expected_weights = np.load(expected_weights_path)
    np.testing.assert_array_almost_equal(som.weights, expected_weights)


@pytest.mark.parametrize("data_seed", [123, 456, 789])
def test_expected_weights_diff_data(data_seed):
    """
    Regression test for training the SOM with different datasets.

    Parameters:
        data_seed (int): Seed to generate different datasets.

    This test compares the SOM's trained weights with expected weights for datasets generated with
    different random seeds.
    """

    np.random.seed(data_seed)
    test_data = np.random.rand(100, INPUT_DIM)
    som = SOMFast(GRID_SIZE, INPUT_DIM)
    som.train(test_data, n_iterations=10, lr=0.1)
    expected_weights_path = this_dir.joinpath(
        f"expected_outputs/expected_weights_diff_data_{data_seed}.npy"
    )
    expected_weights = np.load(expected_weights_path)
    np.testing.assert_array_almost_equal(som.weights, expected_weights)
