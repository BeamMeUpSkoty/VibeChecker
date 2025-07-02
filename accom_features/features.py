from typing import Dict, List, Union
import numpy as np
from accom_features.accom_config import AccomConfig

class AccommodationFeatures:
    """
    Encapsulates synchrony, convergence, state-sequencing, and duration summaries
    for a pair of feature time series A and B.
    """
    # State ID categories
    _SYNC      = {2, 4}
    _ASYN      = {5, 7}
    _CONV      = {3, 4}
    _DIV       = {6, 7}
    _CATEGORIES = {
        "synchrony":   _SYNC,
        "asynchrony":  _ASYN,
        "convergence": _CONV,
        "divergence":  _DIV,
    }

    def __init__(
        self,
        values_A: np.ndarray,
        values_B: np.ndarray,
        cfg: AccomConfig
    ) -> None:
        self.A = values_A
        self.B = values_B
        self.cfg = cfg
        self._cache: Dict[str, Union[float, np.ndarray]] = {}

    def _pearson(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0,1])

    def get_convergence(self) -> float:
        """Global convergence: PearsonCorr(|A-B|, time index)."""
        if 'conv' not in self._cache:
            d = np.abs(self.A - self.B)
            t = np.arange(d.shape[0])
            self._cache['conv'] = self._pearson(d, t)
        return self._cache['conv']  # type: ignore

    def _turn_synchrony(self) -> float:
        """Turn-level static synchrony: PearsonCorr(A[:-1], B[1:])."""
        if 'sync_turn' not in self._cache:
            self._cache['sync_turn'] = self._pearson(self.A[:-1], self.B[1:])
        return self._cache['sync_turn']  # type: ignore

    def _dynamic_synchrony(self) -> (np.ndarray, np.ndarray):
        """Sliding-window synchrony: returns (r_values, r_times)."""
        key = 'sync_dyn'
        if key not in self._cache:
            window, hop = self.cfg.window, self.cfg.hop
            rs: List[float] = []
            times: List[float] = []
            n = len(self.A)
            for i in range(0, n - window + 1, hop):
                winA = self.A[i:i+window]
                winB = self.B[i:i+window]
                rs.append(self._pearson(winA, winB))
                mid = (i + window/2) * self.cfg.frame_duration
                times.append(mid)
            self._cache[key] = (np.array(rs), np.array(times))
        return self._cache[key]  # type: ignore

    def get_synchrony(self) -> Union[float, Dict[str, np.ndarray]]:
        """Dispatch based on cfg.synchrony_mode."""
        mode = self.cfg.synchrony_mode
        if mode == 'turn':
            return self._turn_synchrony()
        dyn_r, dyn_t = self._dynamic_synchrony()
        if mode == 'dynamic':
            return {'r_values': dyn_r, 'r_times': dyn_t}
        # combined
        static_r = self._turn_synchrony()
        return {'r_values': dyn_r, 'r_times': dyn_t, '_static': static_r}

    def get_state_sequence(self) -> np.ndarray:
        """Compute state ID per window using synchrony & convergence."""
        if 'states' not in self._cache:
            rs, _ = self._dynamic_synchrony()
            seq = np.zeros_like(rs, dtype=int)
            # default: maintenance (ID=1)
            seq[:] = 1
            # synchrony-only
            seq[rs >= self.cfg.thresh] = 2
            # divergence-only
            seq[rs <= -self.cfg.thresh] = 6
            # convergence-only overrides
            conv_r = self.get_convergence()
            if conv_r < 0:
                seq[seq == 2] = 4  # sync+conv
            if conv_r > 0:
                seq[seq == 6] = 7  # async+div
            self._cache['states'] = seq
        return self._cache['states']  # type: ignore

    def compute_duration_features(self) -> Dict[str, float]:
        """Total and average-bout durations for each category."""
        states = self.get_state_sequence()
        step = self.cfg.hop * self.cfg.frame_duration
        def runs(mask: np.ndarray) -> List[int]:
            out: List[int] = []
            cnt = 0
            for val in mask:
                if val:
                    cnt += 1
                elif cnt:
                    out.append(cnt)
                    cnt = 0
            if cnt:
                out.append(cnt)
            return out

        feats: Dict[str, float] = {}
        for name, ids in self._CATEGORIES.items():
            mask = np.isin(states, list(ids))
            lengths = runs(mask)
            durs = [l * step for l in lengths]
            feats[f'total_{name}'] = sum(durs)
            feats[f'avg_{name}_bout'] = float(np.mean(durs)) if durs else 0.0
        return feats
