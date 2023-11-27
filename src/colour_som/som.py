import logging
from abc import ABC, abstractmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SOMBase(ABC):
    """Base class for NumPy-based Self-Organising Maps (SOMs).

    This abstract class defines the structure and functionalities for SOMs.
    """

    def __init__(self, grid_size: Tuple[int, int], input_dim: int) -> None:
        """Initializes the SOM with the given grid size and input dimension.

        Args:
            grid_size (tuple): A tuple (width, height) specifying the dimensions of the grid.
            input_dim (int): The dimensionality of the input space.
        """

        self.grid_width, self.grid_height = grid_size
        self.input_dim = input_dim

        if self.grid_width <= 0 or self.grid_height <= 0 or self.input_dim <= 0:
            raise ValueError("Grid dimensions and input dimension must be positive integers.")

        # initialise weights randomly
        self.weights = np.random.rand(self.grid_width, self.grid_height, self.input_dim)

    @abstractmethod
    def find_bmu(self, input_vec: np.ndarray) -> Tuple[int, int]:
        """Abstract method to find the Best Matching Unit (BMU) for a given input vector.

        Args:
            input_vec (numpy.ndarray): An input vector.

        Returns:
            tuple: The grid coordinates of the BMU.
        """

        pass

    @abstractmethod
    def update_weights(
        self,
        input_vec: np.ndarray,
        bmu_idx: Tuple[int, int],
        iteration: int,
        lr: float,
        sigma: float,
        time_constant: float,
    ) -> None:
        """Abstract method to update the weights of the SOM.

        Args:
            input_vec (numpy.ndarray): Input vector.
            bmu_idx (Tuple[int, int]): Coordinates of the BMU.
            iteration (int): Current iteration number.
            lr (float): Learning rate.
            sigma (float): Sigma value for the neighbourhood function.
            time_constant (float): Time constant for the learning rate decay.
        """
        pass

    def load_model(self, filename: str) -> None:
        """Loads SOM's weights from a file.

        Args:
            filename (str): The filename from which to load the weights.
        """
        self.weights = np.load(filename)
        logging.info(f"SOM model loaded from {filename}")

    def save_model(self, filename: str) -> None:
        """Saves the SOM's weights to a file.

        Args:
            filename (str): The filename to save the weights.
        """
        np.save(filename, self.weights)
        logging.info(f"SOM model saved to {filename}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Finds the BMUs for a set of input vectors.

        Args:
            data (np.ndarray): The input vectors.

        Returns:
            np.ndarray: The grid coordinates of the BMUs.
        """
        bmus = np.array([self.find_bmu(vec) for vec in data])
        return bmus

    def quantization_error(self, data: np.ndarray) -> float:
        """Calculates the quantization error of the SOM.

        Args:
            data (numpy.ndarray): Input data set.

        Returns:
            float: The quantization error.
        """
        # compute dist from each data point to all som nodes
        distances = np.linalg.norm(data[:, np.newaxis, np.newaxis, :] - self.weights, axis=3)

        # find indices of closest nodes (i.e. bmu)
        min_indices = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

        # get distances to the bmus for each data point
        bmu_distances = distances[min_indices]

        # return mean dist to the bmu across all data points
        return np.mean(bmu_distances)

    def train(self, data: np.ndarray, n_iterations: int = 100, lr: float = 0.1) -> None:
        """Trains the SOM using the given data.

        Args:
            data (numpy.ndarray): Input data set.
            n_iterations (int): Number of iterations for training. Defaults to 100.
            lr (float): Initial learning rate. Defaults to 0.1.
        """

        logging.info("Starting SOM training.")

        # normalise data
        data = (data - data.min()) / (data.max() - data.min())

        # calc initial sigma as half of grid's largest dim
        sigma = max(self.grid_width, self.grid_height) // 2

        # time constant for adjusting lr and sigma
        time_constant = n_iterations / np.log(sigma)

        # iterate over n iterations
        for i in tqdm(range(n_iterations)):
            # current input vector is randomly selected
            # 1. more likely to cover diverse range of input space
            # 2. prevent SOM overfocusing on patterns or specific areas of input space
            input_vec = data[np.random.randint(len(data))]
            bmu_idx = self.find_bmu(input_vec)
            current_lr, current_sigma = self.update_weights(
                input_vec, bmu_idx, i, lr, sigma, time_constant
            )

            # every 100 iterations, dump progress
            # could probably put a early stopping mechanic here
            # 1. norm of the diff b/w current and prev weights + convergence threshold
            # 2. monitor quant error
            # 3. lr threshold
            # etc.
            if i % 100 == 0:
                # could also calculate topological error
                q_error = self.quantization_error(data)
                logging.info(
                    f"iteration {i}, learning rate: {current_lr:.4f}, sigma: {current_sigma:.4f},"
                    f" quantization error: {q_error:.4f}"
                )

        logging.info("Training completed.")

    def visualise(self, filename=None) -> None:
        """visualises the weight matrix of the SOM."""

        plt.imshow(self.weights)

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


class SOMSlow(SOMBase):
    """Slow implementation of a Self-Organizing Map.

    This subclass of SOMBase provides specific implementations for the
    find_bmu and update_weights methods using explicit loops, which is
    slower but eays to read.
    """

    def find_bmu(self, input_vec: np.ndarray) -> Tuple[int, int]:
        """Finds the Best Matching Unit for a given input vector.

        Args:
            input_vec (numpy.ndarray): Input vector.

        Returns:
            Tuple[int, int]: Grid coordinates of the BMU.
        """

        bmu_idx = None
        min_dist = float("inf")

        # loop thru all the nodes
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                weight_vec = self.weights[i, j]
                distance = np.linalg.norm(input_vec - weight_vec)

                # update bmu if current node is closer to input
                if distance < min_dist:
                    min_dist = distance
                    bmu_idx = (i, j)

        return bmu_idx

    def update_weights(
        self,
        input_vec: np.ndarray,
        bmu_idx: Tuple[int, int],
        iteration: int,
        lr: float,
        sigma: float,
        time_constant: float,
    ) -> None:
        """Updates the weights of the SOM for a given input vector and returns the updated learning rate and sigma.

        Args:
            input_vec (numpy.ndarray): Input vector.
            bmu_idx (Tuple[int, int]): Coordinates of the BMU.
            iteration (int): Current iteration number.
            lr (float): Learning rate.
            sigma (float): Sigma value for the neighbourhood function.
            time_constant (float): Time constant for the learning rate decay.

        Returns:
            Tuple[float, float]: Updated learning rate and sigma.
        """

        # update sigma and lr using exponential decay
        # could also use linear or step decay, but rapid initial lr slow fine-tuning is fine for
        # mapping colours
        sigma = sigma * np.exp(-iteration / time_constant)
        lr = lr * np.exp(-iteration / time_constant)

        # iterate over all nodes in the grid
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # calculate the distance of the node from the bmu
                dist = np.linalg.norm(np.array(bmu_idx) - np.array((i, j)))

                # compute the influence of the bmu on the node
                radius = np.exp(-(dist**2) / (2 * sigma**2))

                # update weight vector of the node
                self.weights[i, j] += radius * lr * (input_vec - self.weights[i, j])

        return lr, sigma


class SOMFast(SOMBase):
    """Fast implementation of a Self-Organizing Map.

    This subclass of SOMBase uses vectorized operations for the
    find_bmu and update_weights methods.
    """

    def find_bmu(self, input_vec: np.ndarray) -> Tuple[int, int]:
        """Finds the Best Matching Unit for a given input vector.

        Args:
            input_vec (numpy.ndarray): Input vector.

        Returns:
            Tuple[int, int]: Grid coordinates of the BMU.
        """
        # reshape to m x n x i for vectorised distance calc
        input_vec = input_vec.reshape((1, 1, self.input_dim))

        # calc euclidean dist between input and all weight vectors
        distances = np.linalg.norm(self.weights - input_vec, axis=2)

        # find index of the min dist, which is the bmu
        return np.unravel_index(np.argmin(distances, axis=None), distances.shape)

    def update_weights(
        self,
        input_vec: np.ndarray,
        bmu_idx: Tuple[int, int],
        iteration: int,
        lr: float,
        sigma: float,
        time_constant: float,
    ) -> None:
        """Updates the weights of the SOM using vectorized operations.

        Args:
            input_vec (numpy.ndarray): Input vector.
            bmu_idx (Tuple[int, int]): Coordinates of the BMU.
            iteration (int): Current iteration number.
            lr (float): Learning rate.
            sigma (float): Sigma value for the neighbourhood function.
            time_constant (float): Time constant for the learning rate decay.

        Returns:
            Tuple[float, float]: Updated learning rate and sigma.
        """

        # update sigma and lr using exponential decay
        sigma = sigma * np.exp(-iteration / time_constant)
        lr = lr * np.exp(-iteration / time_constant)

        # create a 2D grid of coordinates, x and y are matrices of shape m x n
        x, y = np.indices((self.grid_width, self.grid_height))

        # calculate squared Euclidean distance of each node from the BMU
        euclidean_dist = (x - bmu_idx[0]) ** 2 + (y - bmu_idx[1]) ** 2
        influence = np.exp(-euclidean_dist / (2 * sigma**2))

        # influence needs to be applied element-wise, align shapes for rbaodcasting to m x n x i
        influence = influence.reshape((self.grid_width, self.grid_height, 1))
        self.weights += influence * lr * (input_vec - self.weights)

        return lr, sigma
