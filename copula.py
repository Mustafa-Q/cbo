import numpy as np
from scipy.stats import norm

# ----------------------------
# Default Probability Helper
# ----------------------------
def score_to_default_rate(score: int) -> float:
    if score > 700:
        return 0.02  
    elif score > 650:
        return 0.05  
    else:
        return 0.08  

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