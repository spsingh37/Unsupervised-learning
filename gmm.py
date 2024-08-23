import numpy as np
from typing import NamedTuple, Union, Literal


def hello():
    print('Hello from gmm.py!')


class GMMState(NamedTuple):
    """Parameters to a GMM Model."""
    pi: np.ndarray  # [K]
    mu: np.ndarray  # [K, d]
    sigma: np.ndarray  # [K, d, d]


def train_gmm(train_data: np.ndarray,
              init_pi: np.ndarray,
              init_mu: np.ndarray,
              init_sigma: np.ndarray,
              *,
              num_iterations: int = 50,
              ) -> GMMState:
    """Fit a GMM model.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        init_pi: The initial value of pi. Shape (K, )
        init_mu: The initial value of mu. Shape (K, d)
        init_sigma: The initial value of sigma. Shape (K, d, d)
        num_iterations: Run EM (E-steps and M-steps) for this number of
            iterations.

    Returns:
        A GMM parameter after running `num_iterations` number of EM steps.
    """
    # Sanity check
    N, d = train_data.shape
    K, = init_pi.shape
    assert init_mu.shape == (K, d)
    assert init_sigma.shape == (K, d, d)

    ###########################################################################
    # Implement EM algorithm for learning GMM.
    ###########################################################################
    # TODO: Add your implementation.
    # Feel free to add helper functions as much as needed.
    pi, mu, sigma = init_pi.copy(), init_mu.copy(), init_sigma.copy()

    #######################################################################
    # ~~START DELETE~~
    for i in range(num_iterations):
        # E-step
        gamma, log_totals = compute_gamma(pi, mu, sigma, train_data) # [K x Ndata]
        num_points_in_clusters = gamma.sum(axis=1, keepdims=True) # N_k [K x Ndata].sum() -> [K x 1]

        # M-step
        # See p.46 of the lecture 15

        ### \pi
        pi = (num_points_in_clusters / N).reshape([K])

        ### \mu
        gamma_x_sum = np.matmul(gamma, train_data) # [K x Ndata ] x [Ndata x dim] = [K x dim]
        mu = gamma_x_sum / num_points_in_clusters  # [K x dim]

        log_likelihood = np.sum(log_totals)
        print(f'Iteration {i:2d}: log-likelihood={log_likelihood:6.2f}')

        ### \Sigma (slow version)
        #for index in range(K):
        #    sum_mat = 0.
        #    for data_ind, data in enumerate(train_data):
        #        x_zero_meaned = np.expand_dims(data - mu[index], 1)
        #        sum_mat += gamma[index, data_ind] * np.matmul(x_zero_meaned, x_zero_meaned.transpose())
        #    sigma[index] = sum_mat / num_points_in_clusters[index]  # [dim x dim] / [1] = [dim x dim]

        ### \Sigma
        # fast-version
        for k in range(K):
            N_k = np.sum(gamma[k])
            sigma[k] = (gamma[k][..., np.newaxis] * (train_data - mu[k])).T @ (train_data - mu[k]) / N_k
            #           -------------------------   --------------------      -------------------
            #                   (N, 1)                      (N, d)                  (N, d)

    # ~~END DELETE~~
    #######################################################################

    return GMMState(pi, mu, sigma)


# ~~START DELETE NO TODO~~
def compute_gamma(pi, mu, sigma, data):
    """Compute gamma and log-likelihood for E-step.

    Returns:
        gamma, shape [num_clusters, num_data]
        log-likelihood, shape [num_data]
    """
    from scipy.stats import multivariate_normal
    from scipy.special import logsumexp

    num_data, ndim = data.shape
    K = mu.shape[0]   # number of GMM clusters
    log_normpdf = np.zeros((K, num_data))
    for index in range(K):
        # log(normpdf): [K x Ndata]. to ensure numerical stability
        # Note that the raw likelihood (pdf) can be too low for some poor values of mu and sigma
        log_normpdf[index, :] = multivariate_normal(mu[index], sigma[index]).logpdf(data)

    # \log(\pi_k * N(...))  [K x Ndata]
    log_pi_normpdf = np.log(pi + 1e-10).reshape(-1, 1) + log_normpdf
    # \log \sum_j [ \pi_j N(...) ]
    log_totals = logsumexp(log_pi_normpdf, axis=0, keepdims=True)

    # p(zk=1 | x)) # [Ncentroid x Ndata]
    gamma = np.exp(log_pi_normpdf - log_totals)  # [K x Ndata]
    return gamma, log_totals
# ~~END DELETE~~


def compress_image(image: np.ndarray, gmm_model: GMMState) -> np.ndarray:
    """Compress image by mapping each pixel to the mean value of a
    Gaussian component (hard assignment).

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        gmm_model: type GMMState. A GMM model parameters.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K = gmm_model.mu.shape[0]

    ##########################################################################
    # ~~START DELETE~~
    test_data = image.reshape(-1, C)
    pi, mu, sigma = gmm_model

    gamma, _ = compute_gamma(pi, mu, sigma, test_data)
    assignment = np.argmax(gamma, axis=0)

    # Note: it's possible to optimize and vectorize the following
    # Note: to do 'soft assignment', you can do `compressed_data = gamma.T @ mu`
    compressed_data = test_data.copy()
    for index in range(K):
        mask = (assignment == index)
        compressed_data[mask, :] = mu[index]

    compressed_image = np.rint(compressed_data.reshape([H, W, C])).astype(np.uint8)
    # ~~END DELETE~~
    ##########################################################################

    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
