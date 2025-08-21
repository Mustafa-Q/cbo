import numpy as np
from scipy.stats import norm

import numpy as np
from scipy.stats import norm, t as student_t

# ----------------------------
# Gaussian Copula Defaults
# ----------------------------
def generate_correlated_defaults(n_loans, default_probs, rho=0.2, seed=None):
    """
    Generate correlated default flags using a Gaussian copula.
    """
    if seed is not None:
        np.random.seed(seed)

    corr_matrix = rho * np.ones((n_loans, n_loans)) + (1 - rho) * np.eye(n_loans)
    L = np.linalg.cholesky(corr_matrix)

    Z_indep = np.random.normal(size=(n_loans,))
    Z_corr = L @ Z_indep

    thresholds = norm.ppf(default_probs)
    return Z_corr < thresholds


# ----------------------------
# t-Copula Defaults (Tail Dependence)
# ----------------------------
def generate_t_copula_defaults(n_loans, default_probs, rho=0.2, df=3, seed=None):
    """
    Generate correlated defaults using a t-copula with tail dependence.
    """
    if seed is not None:
        np.random.seed(seed)

    corr_matrix = rho * np.ones((n_loans, n_loans)) + (1 - rho) * np.eye(n_loans)
    L = np.linalg.cholesky(corr_matrix)

    Z_indep = student_t.rvs(df, size=n_loans)
    Z_corr = L @ Z_indep

    U = student_t.cdf(Z_corr, df)  # Convert to uniforms
    return U < default_probs

# ----------------------------
# Correlated Default Generator
# ----------------------------
def generate_correlated_defaults(n_loans, default_probs, rho=0.2, seed=None):
    """
    Generate correlated default flags using a Gaussian copula.

    Parameters:
        n_loans (int): Number of loans
        default_probs (list or np.array): Marginal default probabilities for each loan
        rho (float): Correlation coefficient (between 0 and 1)
        seed (int or None): Random seed for reproducibility

    Returns:
        np.array of bools: True means the loan defaulted
    """
    if seed is not None:
        np.random.seed(seed)

    # Create correlation matrix
    corr_matrix = rho * np.ones((n_loans, n_loans)) + (1 - rho) * np.eye(n_loans)
    L = np.linalg.cholesky(corr_matrix)

    # Generate independent standard normals
    Z_indep = np.random.normal(size=(n_loans,))
    Z_corr = L @ Z_indep

    # Determine default based on inverse CDF (quantile function)
    thresholds = norm.ppf(default_probs)
    return Z_corr < thresholds

