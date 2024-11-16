import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize


def negative_log_likelihood(params, data):
    """
    Negative log-likelihood function for the Gamma distribution.
    :param params: Array of parameters [k (shape), theta (scale)].
    :param data: Array of observed data points.
    :return: Negative log-likelihood value.
    """
    k, theta = params[0], params[1]
    if k <= 0 or theta <= 0:  # Constraints: k > 0, theta > 0
        return np.inf

    n = len(data)
    log_likelihood = (
            n * (k * np.log(theta) - gammaln(k)) +
            (k - 1) * np.sum(np.log(data)) - (1 / theta) * np.sum(data)
    )
    return -log_likelihood  # Negative because we are minimizing


def estimate_gamma_parameters(data):
    """
    Estimate Gamma distribution parameters using Maximum Likelihood Estimation.
    :param data: Array of observed data points.
    :return: Estimated parameters k (shape) and theta (scale).
    """
    # Calculate sample mean and variance for better initial guesses
    sample_mean = np.mean(data)
    sample_variance = np.var(data)
    initial_k = sample_mean ** 2 / sample_variance
    initial_theta = sample_variance / sample_mean

    # Initial guess for optimization
    initial_guess = [initial_k, initial_theta]

    # Minimize the negative log-likelihood
    result = minimize(
        negative_log_likelihood,
        initial_guess,
        args=(data,),
        method="SLSQP",
        bounds=[(1e-3, None), (1e-3, None)],  # Ensure k > 0, theta > 0
    )

    if result.success:
        estimated_k, estimated_theta = result.x
        return estimated_k, estimated_theta
    else:
        raise RuntimeError("Optimization failed: " + result.message)


if __name__ == "__main__":
    # Load data from CSV file
    import pandas as pd

    data = pd.read_csv("part 2. input.csv").iloc[:, 0].values

    # Estimate Gamma parameters
    estimated_k, estimated_theta = estimate_gamma_parameters(data)

    # Print estimated parameters
    print(estimated_k, estimated_theta)


    
