from accomodation_types.base_accomodation import BaseAccommodation
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


class TurnLevelProsodicAccomodation(BaseAccommodation):
    """
    Turn-Taking Prosodic Accommodation.

    At each turn exchange i:
      - Take the i-th utterance of speaker A and the i-th utterance of speaker B,
        extract only the requested features from each chunk, store as (A_i, B_i).
    Convergence: PearsonCorr(|A_i – B_i|, i).
    Synchrony: PearsonCorr(A_{i-1}, B_i) over all valid i ≥ 1.
    """

    def __init__(
        self,
        audio_path: str,
        transcript_csv: str,
        requested_features: list[str] = None,
        verbose: bool = False,
    ):
        """
        :param audio_path: path to mixed-speaker WAV.
        :param transcript_csv: path to CSV with columns [start,end,text,speaker].
        :param requested_features: list of feature names to extract (e.g. ["mean_f0","mean_intensity"]).
                                   If None, defaults to ["mean_f0","sd_f0","mean_intensity","syllables_per_second"].
        :param verbose: If True, pass to AudioFeatures.extract(...) for each chunk.
        """
        super().__init__(audio_path, transcript_csv, requested_features=requested_features, verbose=verbose)

        # Identify speaker IDs
        self.speaker_A = self.speaker_ids[0]
        self.speaker_B = self.speaker_ids[1]

        # Turn-level uses one utterance per speaker per “exchange index.”
        self.utts_A = self.utts_by_speaker[self.speaker_A]
        self.utts_B = self.utts_by_speaker[self.speaker_B]
        # cache for per-turn features
        self._accom_cache: Dict[str, np.ndarray] = {}

    def get_accommodation(self) -> dict[str, np.ndarray]:
        """
        Returns a dict mapping each feature → np.ndarray of shape (n_exchanges, 2),
        where [:,0] = speaker A’s value, [:,1] = speaker B’s value.

        We assume turn-exchange i corresponds to the i-th utterance of A and B.
        """
        # return cached if exists
        if self._accom_cache:
            return self._accom_cache

        n_exchanges = min(len(self.utts_A), len(self.utts_B))
        feat_names = self.requested_features

        # Initialize arrays: one (n_exchanges×2) array per feature
        accom = {f: np.zeros((n_exchanges, 2), dtype=float) for f in feat_names}

        for idx in range(n_exchanges):
            utt_A = self.utts_A[idx]
            utt_B = self.utts_B[idx]

            # Load raw audio segments for speaker A’s idx-th utterance
            startA, endA = utt_A["start"], utt_A["end"]
            chunk_A = self._get_speaker_window_chunk(self.speaker_A, startA, endA)

            # Similarly for speaker B
            startB, endB = utt_B["start"], utt_B["end"]
            chunk_B = self._get_speaker_window_chunk(self.speaker_B, startB, endB)

            # Extract exactly the requested features from each chunk
            feats_A = self._wrap_and_extract(chunk_A)  # dict: feature→value
            feats_B = self._wrap_and_extract(chunk_B)

            # Fill the arrays
            for f in feat_names:
                accom[f][idx, 0] = feats_A.get(f, 0.0)
                accom[f][idx, 1] = feats_B.get(f, 0.0)

        self._accom_cache = accom
        return accom

    def get_convergence(self) -> dict[str, float]:
        """
        For each requested feature f:
          - Let A_series = accom[f][:,0], B_series = accom[f][:,1], length = n_exchanges.
          - Let d = |A_series – B_series|, t = [0, 1, …, n_exchanges-1].
          - Return PearsonCorr(d, t).
        """
        accom = self.get_accommodation()
        results: Dict[str, float] = {}
        # vector of exchange indices
        t = np.arange(next(iter(accom.values())).shape[0])
        for f, pairs in accom.items():
            d = np.abs(pairs[:,0] - pairs[:,1])
            if d.size < 2 or np.std(d)==0 or np.std(t)==0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(d, t)[0,1])
            results[f] = corr
        return results

    def _turn_synchrony(self) -> dict[str, float]:
        '''
        Turn-Taking synchrony: For each feature f:
          - Let A_prev = [A_0, A_1, …, A_{n-2}], B_curr = [B_1, …, B_{n-1}].
          - Return PearsonCorr(A_prev, B_curr). If n_exchanges ≤ 1, return 0.0.
        '''
        accom = self.get_accommodation()
        results: Dict[str, float] = {}
        for f in self.requested_features:
            pairs = accom[f]
            if pairs.shape[0] <= 1:
                results[f] = 0.0
            else:
                results[f] = self._pearsonr(pairs[:-1,0], pairs[1:,1])
        return results

    def _dynamic_synchrony(self, window, hop):
        accom = self.get_accommodation()  # dict: feat → Nx2 array
        results = {}
        for feat, pair_arr in accom.items():
            rs = []
            for idx, i in enumerate(range(0, len(pair_arr) - window + 1, hop)):
                a_win = pair_arr[i:i + window, 0]
                b_win = pair_arr[i:i + window, 1]
                r = self._pearsonr(a_win, b_win)
                if idx < 3:  # only print first 3 windows
                    print(f"[DEBUG][{feat}] window {idx}: r = {r:.4f}")
                rs.append(r)
            results[feat] = np.array(rs)
        return results

    def get_synchrony(self) -> dict:
        mode = getattr(self, "synchrony_mode", "turn")
        if mode == "turn":
            print(f"[DEBUG][turn] using _turn_synchrony()")
            return self._turn_synchrony()
        elif mode == "dynamic":
            print(f"[DEBUG][dynamic] window={self.win_frames}, hop={self.hop_frames}")
            raw = self._dynamic_synchrony(self.win_frames, self.hop_frames)
            return {feat: {"r_values": raw[feat]}
                    for feat in self.requested_features}
        elif mode == "combined":
            print(f"[DEBUG][combined] combining static+dynamic")
            static = self._turn_synchrony()
            print(f"[DEBUG][static] mean_f0 r = {static['mean_f0']:.4f}")
            dyn = self._dynamic_synchrony(self.win_frames, self.hop_frames)
            combined = {
                f: (static[f] + float(np.mean(dyn[f]))) / 2
                for f in self.requested_features
            }
            return combined
        else:
            raise ValueError(f"Unknown synchrony_mode {mode!r}")
