



import numpy as np

_N_STATES   = 5
_STAGE_W    = np.array([1.0, 1.0, 0.0, 0.0, 0.0])    # position only, every t
_TERMINAL_W = np.array([1.0, 1.0, 0.0, 0.3, 0.3])    # position + "stop here" at the end
_TERMINAL_SCALE = 80.0

def _weights(r: np.ndarray) -> np.ndarray:
    n = _N_STATES
    N1 = len(r) // n
    w = np.tile(_STAGE_W, N1)
    w[-n:] = _TERMINAL_W * _TERMINAL_SCALE
    return w

def Cx(r: np.ndarray, r_target: np.ndarray, alpha: float) -> float:
    """State cost: weighted squared distance to target trajectory."""
    error = r - r_target
    return alpha * np.sum(_weights(r) * error ** 2)

def Cx_grad(r: np.ndarray, r_target: np.ndarray, alpha: float) -> np.ndarray:
    """Analytic gradient of Cx wrt r."""
    return 2.0 * alpha * _weights(r) * (r - r_target)


def Cu(u: np.ndarray, alpha: float) -> float:
    """Control cost: weighted squared control effort."""
    return (1.0 - alpha) * np.sum(u ** 2)


def L(
    x        : np.ndarray,
    u        : np.ndarray,
    r        : np.ndarray,
    v        : np.ndarray,
    r_target : np.ndarray,
    P        : float,
    alpha    : float,
) -> float:
    
    return (
        Cx(r, r_target, alpha)
        + Cu(u, alpha)
        + (P / 2) * (
            np.linalg.norm(x - r + v) ** 2
            - np.linalg.norm(v)       ** 2
        )
    )