import numpy as np
import pytest

from colour_som import SOMFast

GRID_SIZE = (10, 10)
INPUT_DIM = 3
np.random.seed(34)
TEST_DATA = np.random.rand(100, INPUT_DIM)


def test_initialization():
    """
    Test the initialization of the SOMFast class.

    Ensures that the SOM is initialized with the correct grid dimensions, input dimension,
    and weight matrix shape.
    """

    som = SOMFast(GRID_SIZE, INPUT_DIM)
    assert som.grid_width == GRID_SIZE[0]
    assert som.grid_height == GRID_SIZE[1]
    assert som.input_dim == INPUT_DIM
    assert som.weights.shape == (*GRID_SIZE, INPUT_DIM)


def test_find_bmu():
    """
    Test the Best Matching Unit (BMU) finding functionality.

    Verifies that the find_bmu method returns a tuple of two integers (the coordinates of the BMU)
    for a given input vector.
    """

    som = SOMFast(GRID_SIZE, INPUT_DIM)
    bmu = som.find_bmu(np.random.rand(INPUT_DIM))
    assert isinstance(bmu, tuple) and len(bmu) == 2


def test_quantization_error():
    """
    Test the calculation of the quantization error.

    Checks that the method quantization_error returns a float value and that this value is
    non-negative, ensuring that the quantization error calculation is working as expected.
    """

    som = SOMFast(GRID_SIZE, INPUT_DIM)
    error = som.quantization_error(TEST_DATA)
    assert isinstance(error, float)
    assert error >= 0


def test_invalid_initialization():
    """
    Test the initialization of the SOMFast class with invalid parameters.

    Verifies that the SOM raises a ValueError when initialized with negative or zero values
    for grid dimensions or input dimension.
    """

    with pytest.raises(ValueError):
        SOMFast((-1, 10), INPUT_DIM)
    with pytest.raises(ValueError):
        SOMFast((10, -1), INPUT_DIM)
    with pytest.raises(ValueError):
        SOMFast(GRID_SIZE, 0)


def test_find_bmu_known_weights():
    """
    Test the BMU finding with known weights.

    Sets a known weight matrix and tests if find_bmu method returns the correct BMU
    for a predefined input vector.
    """

    som = SOMFast((2, 2), INPUT_DIM)
    som.weights = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]])
    bmu = som.find_bmu(np.array([0.2, 0.3, 0.4]))
    assert bmu == (0, 0)


def test_update_weights():
    """
    Test the weight update functionality.

    Ensures that the weights of the SOM are updated when the update_weights method is called.
    This is verified by checking that the weight matrix changes after the update.
    """

    som = SOMFast((2, 2), INPUT_DIM)
    original_weights = np.copy(som.weights)
    som.update_weights(np.array([0.5, 0.5, 0.5]), (1, 1), 0, 0.1, 1.0, 1.0)
    assert not np.array_equal(original_weights, som.weights)


def test_training_improves_quantization_error():
    """
    Test the effectiveness of the SOM training.

    Checks whether training the SOM with a dataset reduces the quantization error,
    indicating successful learning.
    """

    som = SOMFast(GRID_SIZE, INPUT_DIM)
    initial_error = som.quantization_error(TEST_DATA)
    som.train(TEST_DATA, n_iterations=10)
    final_error = som.quantization_error(TEST_DATA)
    assert final_error < initial_error
