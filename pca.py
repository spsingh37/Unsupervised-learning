from typing import Tuple
import numpy as np


def hello_world():
    print("Hello world")


def train_PCA(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run PCA on the data.

    Input:
        data: A numpy array of shape [N, d], where N is the number of data
            points and d is the dimension of each data point.
            We assume the data has full rank.

    Returns: A tuple of (U, eigenvalues)
        U: The U matrix, whose column vectors are principal components
            (i.e., eigenvectors) in the order of decreasing variance.
        eigenvalues:
            An array (or list) of all eigenvalues sorted in a decreasing order.
    """
    if len(data.shape) != 2:
        raise ValueError("Invalid shape of data; did you forget flattening?")
    N, d = data.shape

    #######################################################
    ###              START OF YOUR CODE                 ###
    #######################################################
    centered_data = data - np.mean(data, axis=0, keepdims=True)
    assert np.allclose(centered_data.mean(axis=0), np.zeros(d))

    # Important: bias=False is default in np.cov, which divides by "N - 1".
    # The correct implementation is to divide by "N", where bias=True should
    # be used.
    # In the future (2024), guide students to not use np.cov but use the
    # definition directly. Or explicitly give a hint that bias=True is needed.
    # To future GSIs: please make the 1 / (N-1) one incorrect (e.g., -0.5 pts)
    # by improving the test cases, specifically detecting this failure mode.
    cov = np.cov(centered_data.T, bias=True)

    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    indices = np.argsort(-eigen_vals)
    U = eigen_vecs[:, indices]
    eigenvalues = eigen_vals[indices]
    #######################################################
    ###                END OF YOUR CODE                 ###
    #######################################################

    return U, eigenvalues
