import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def pearsonr_vec(winA: np.ndarray, winB: np.ndarray) -> np.ndarray:
    Az = (winA - winA.mean(axis=1, keepdims=True)) / winA.std(axis=1, keepdims=True)
    Bz = (winB - winB.mean(axis=1, keepdims=True)) / winB.std(axis=1, keepdims=True)
    return np.einsum('ij,ij->i', Az, Bz) / (winA.shape[1] - 1)

def sliding_windows(series: np.ndarray, window: int, hop: int) -> np.ndarray:
    all_wins = sliding_window_view(series, window)
    return all_wins[::hop]
