from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import numpy as np
from accom_features.accom_config import AccomConfig
from accom_features.utils import sliding_windows, pearsonr_vec
from accom_features.features import AccommodationFeatures
from itertools import groupby

class FeatureStrategy(ABC):
    def __init__(self, A: np.ndarray, B: np.ndarray, cfg: AccomConfig):
        self.A, self.B, self.cfg = A, B, cfg

    @abstractmethod
    def compute(self) -> Any:
        """Return the primary metric (scalar or dict)."""

    @abstractmethod
    def states(self) -> np.ndarray:
        """Return per-window state IDs."""

    @abstractmethod
    def durations(self) -> Dict[str, float]:
        """Return total/avg‐bout durations per category."""

class TurnSynchrony(FeatureStrategy):
    def compute(self) -> float:
        return float(np.corrcoef(self.A[:-1], self.B[1:])[0,1])
    def states(self): raise NotImplementedError
    def durations(self): raise NotImplementedError

class DynamicSynchrony(FeatureStrategy):
    def compute(self) -> Dict[str, np.ndarray]:
        winsA = sliding_windows(self.A, self.cfg.window, self.cfg.hop)
        winsB = sliding_windows(self.B, self.cfg.window, self.cfg.hop)
        rs = pearsonr_vec(winsA, winsB)
        times = (np.arange(len(rs)) * self.cfg.hop + self.cfg.window/2) * self.cfg.frame_duration
        return {"r_values": rs, "r_times": times}

    def states(self) -> np.ndarray:
        rs = self.compute()["r_values"]
        s = np.zeros_like(rs, dtype=int)
        s[rs >= self.cfg.thresh] = 2
        s[rs <= -self.cfg.thresh] = 6
        return s

    def durations(self) -> Dict[str, float]:
        from itertools import groupby
        seq = self.states()
        step = self.cfg.hop * self.cfg.frame_duration
        out = {}
        for name, ids in AccommodationFeatures._CATEGORIES.items():
            runs = [sum(1 for _ in grp) for val, grp in groupby(seq) if val in ids]
            durs = [r * step for r in runs]
            out[f"total_{name}"] = sum(durs)
            out[f"avg_{name}"]   = (sum(durs)/len(durs)) if durs else 0.0
        return out
class CombinedSynchrony(FeatureStrategy):
    def compute(self):
        dyn = DynamicSynchrony(self.A, self.B, self.cfg).compute()
        static = TurnSynchrony(self.A, self.B, self.cfg).compute()
        dyn["r_values"]  # reused
        dyn["r_times"]
        dyn["_static"] = static
        return dyn

    def states(self):
        return DynamicSynchrony(self.A, self.B, self.cfg).states()

    def durations(self):
        return DynamicSynchrony(self.A, self.B, self.cfg).durations()

class ConvergenceStrategy(FeatureStrategy):
    def compute(self) -> float:
        d = np.abs(self.A - self.B)
        t = np.arange(len(d))
        return float(np.corrcoef(d, t)[0,1])  # or use pearsonr_vec on 2×N
    def states(self): raise NotImplementedError
    def durations(self): raise NotImplementedError

class ConcurrentFeatureStrategy:
    """
    Summarizes where *all* input features share the same state
    (sync/async/conv/div) in each window.
    """
    def __init__(self,
                 masks_list: List[Dict[str, np.ndarray]],
                 hop_seconds: float):
        """
        masks_list: List of dicts, one per feature, with keys 'sync','async','conv','div'
        hop_seconds: window hop duration in seconds
        """
        self.masks_list = masks_list
        self.hop_seconds = hop_seconds

    def durations(self) -> Dict[str, float]:
        # Use the provided masks list directly
        feature_masks = self.masks_list

        # Build concurrent masks by AND-ing across all features
        all_sync  = np.logical_and.reduce([m['synchrony']  for m in feature_masks])
        all_async = np.logical_and.reduce([m['asynchrony'] for m in feature_masks])
        all_conv  = np.logical_and.reduce([m['convergence']  for m in feature_masks])
        all_div   = np.logical_and.reduce([m['divergence']   for m in feature_masks])

        masks = {
            'synchrony':   all_sync,
            'asynchrony':  all_async,
            'convergence': all_conv,
            'divergence':  all_div,
        }

        return self._summarize_concurrent_states(masks, self.hop_seconds)

    @staticmethod
    def _summarize_concurrent_states(state_masks: Dict[str, np.ndarray],
                                     hop_seconds: float
                                    ) -> Dict[str, float]:
        """
        Given boolean masks per state, return total and average
        durations for each concurrent state.
        """
        results: Dict[str, float] = {}
        for name, mask in state_masks.items():
            # run-length encode True spans
            lengths = [sum(1 for _ in grp)
                       for val, grp in groupby(mask) if val]
            if lengths:
                durations = [L * hop_seconds for L in lengths]
                total = sum(durations)
                avg   = total / len(durations)
            else:
                total = avg = 0.0
            results[f"total_concurrent_{name}"] = total
            results[f"avg_concurrent_{name}"]   = avg
        return results

