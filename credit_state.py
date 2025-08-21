import numpy as np

CREDIT_STATES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

TRANSITION_MATRIX = np.array([
    [0.90, 0.07, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00],  # AAA
    [0.02, 0.88, 0.07, 0.02, 0.01, 0.00, 0.00, 0.00],  # AA
    [0.00, 0.03, 0.85, 0.08, 0.03, 0.01, 0.00, 0.00],  # A
    [0.00, 0.01, 0.04, 0.80, 0.10, 0.04, 0.01, 0.00],  # BBB
    [0.00, 0.00, 0.01, 0.05, 0.78, 0.10, 0.05, 0.01],  # BB
    [0.00, 0.00, 0.00, 0.01, 0.04, 0.80, 0.10, 0.05],  # B
    [0.00, 0.00, 0.00, 0.00, 0.01, 0.05, 0.80, 0.14],  # CCC
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # D
])

def simulate_credit_transition(current_state: str, seed: int = None) -> str:
    """
    Simulate a single transition from current credit state using Markov chain.
    """
    if seed is not None:
        np.random.seed(seed)
    i = CREDIT_STATES.index(current_state)
    return np.random.choice(CREDIT_STATES, p=TRANSITION_MATRIX[i])
