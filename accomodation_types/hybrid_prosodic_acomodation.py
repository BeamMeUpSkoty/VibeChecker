import numpy as np
from accomodation_types.base_accomodation import BaseAccommodation
import matplotlib.pyplot as plt
from typing import Dict, List

class HYBRIDProsodicAcommodation(BaseAccommodation):
    '''
    Hybrid (Utterance‐Sensitive TAMA), uses a single WAV + single CSV.

    - Start with fixed windows [t₀, t₀ + window_len), hop = self.hop.
    - For each speaker separately:
        - Find nearest utterance‐boundary start ≤ t₀,
            nearest utterance‐boundary end   ≥ (t₀ + window_len).
        - This yields [sA, eA] for speaker A, [sB, eB] for speaker B.
        - Extract speaker‐A audio in [sA, eA) and speaker‐B audio in [sB, eB).
    - For each nominal fixed window [t0, t0+window_len):
        - Extend that window to whole‐utterance boundaries separately for A and B.
        - Extract only requested features from each speaker’s extended‐segment.

    Convergence and Synchrony are computed exactly as in TAMA.
        - Convergence: PearsonCorr(|A_t−B_t|, t).
        - Synchrony: Sliding-window Pearson over A_series vs B_series.
    '''

    def __init__(
        self,
        audio_path: str,
        transcript_csv: str,
        requested_features: list[str] = None,
        window_len: float = 20.0,
        hop: float = 10.0,
        verbose: bool = False
    ):
        """
        :param audio_path: path to mixed-speaker WAV.
        :param transcript_csv: path to CSV [start,end,text,speaker].
        :param requested_features: list of features from AudioFeatures.extract().
        :param window_len: nominal window length (sec).
        :param hop: hop size (sec).
        :param verbose: pass to AudioFeatures.extract().
        """

        #Initialize BaseAccommodation
        super().__init__(audio_path, transcript_csv, requested_features=requested_features, verbose=verbose)

        self.window_len = window_len
        self.hop = hop

        # Precompute nominal window starts
        self.window_starts = np.arange(
            0.0, self.duration - self.window_len + 1e-8, self.hop
        )

        # Store speaker IDs and their utterance lists
        self.speaker_A = self.speaker_ids[0]
        self.speaker_B = self.speaker_ids[1]

        # For convenience, grab lists of utterance (start, end) for each speaker
        # Each utt is a dict { 'start':float, 'end':float, 'text':str }
        self.utts_A = self.utts_by_speaker[self.speaker_A]
        self.utts_B = self.utts_by_speaker[self.speaker_B]

        # Cached feature matrix: dict[feature] = np.ndarray shape (nwin, 2)
        self._accom_cache: Dict[str, np.ndarray] = {}

    def _extend_window(self, t0: float, t1: float, utts: list[dict]) -> tuple[float, float]:
        """
        Given a nominal window [t0, t1), find the nearest full‐utterance boundaries:
          new_start = max{ utt['start'] ≤ t0 }
          new_end   = min{ utt['end']   ≥ t1 }
        If none satisfy, fallback to t0 or t1 respectively.
        """
        starts = [u["start"] for u in utts if u["start"] <= t0]
        new_start = max(starts) if starts else t0

        ends = [u["end"] for u in utts if u["end"] >= t1]
        new_end = min(ends) if ends else t1

        return new_start, new_end

    def get_accommodation(self) -> dict[str, np.ndarray]:
        """
        Returns a dict mapping each requested feature → np.ndarray of shape (nwin, 2):
          [:,0] = speaker A’s value, [:,1] = speaker B’s value, over nwin windows.

        For each window index i:
            t0 = self.window_starts[i], t1 = t0 + window_len
            (sA, eA) = _extend_window(t0, t1, self.utts_A)
            (sB, eB) = _extend_window(t0, t1, self.utts_B)
            chunk_A = speaker-A audio in [sA, eA), chunk_B = speaker-B audio in [sB, eB)
            feats_A = _wrap_and_extract(chunk_A), feats_B = _wrap_and_extract(chunk_B)
            Fill accom[f][i,0] and accom[f][i,1]
        """
        # Build cache on first call
        if self._accom_cache:
            return self._accom_cache

        nwin = len(self.window_starts)
        feat_names = self.requested_features

        accom = {f: np.zeros((nwin, 2), dtype=float) for f in feat_names}

        for idx, t0 in enumerate(self.window_starts):
            t1 = t0 + self.window_len

            # compute extended window for speaker A and B
            sA, eA = self._extend_window(t0, t1, self.utts_A)
            sB, eB = self._extend_window(t0, t1, self.utts_B)

            # gather each speaker’s audio in [sA,eA) and [sB,eB)
            chunk_A = self._get_speaker_window_chunk(self.speaker_A, sA, eA)
            chunk_B = self._get_speaker_window_chunk(self.speaker_B, sB, eB)

            # extract requested features
            feats_A = self._wrap_and_extract(chunk_A)
            feats_B = self._wrap_and_extract(chunk_B)

            # 4) fill arrays
            for f in feat_names:
                accom[f][idx, 0] = feats_A.get(f, 0.0)
                accom[f][idx, 1] = feats_B.get(f, 0.0)

        self._accom_cache = accom
        return accom

    def get_convergence(self) -> dict[str, float]:
        """
        Exactly as in TAMA: for each feature, let d_i = |A_i − B_i| and t_i = i.
        Return PearsonCorr(d, t).

        For each feature f:
          - A_series = accom[f][:,0], B_series = accom[f][:,1]
          - d = |A_series − B_series|, t = [0..nwin-1]
          - r = PearsonCorr(d, t). Return { f: r }.
        """
        accom = self.get_accommodation()
        results: Dict[str, float] = {}
        for f, pairs in accom.items():
            # difference series
            d = np.abs(pairs[:, 0] - pairs[:, 1])
            # vectorized correlation with time index
            t = np.arange(len(d))
            if np.std(d) == 0 or np.std(t) == 0:
                results[f] = 0.0
            else:
                corr = np.corrcoef(d, t)[0, 1]
                results[f] = float(corr)
        return results

    def get_synchrony(self, sync_window: int = 5) -> dict[str, np.ndarray]:
        """
        Sliding‐window synchrony over the feature‐streams A[·], B[·].

        For each feat, we have A_i and B_i for i=0..nwin−1.
        For i in 0..(nwin - sync_window):
          compute r_i = PearsonCorr(A[i:i+sync_window], B[i:i+sync_window])
        Return arrays of length (nwin - sync_window + 1) for each feature.

        Sliding-window synchrony: For each feature f:
          - A_series = accom[f][:,0], B_series = accom[f][:,1], length = nwin
          - r_i = PearsonCorr(A[i:i+sync_window], B[i:i+sync_window]) for i=0..nwin-sync_window
          - Return array of length (nwin-sync_window+1).
        """
        accom = self.get_accommodation()
        results: Dict[str, np.ndarray] = {}
        for f, pairs in accom.items():
            A_s = pairs[:, 0]
            B_s = pairs[:, 1]
            nwin = len(A_s)
            if nwin < sync_window:
                results[f] = np.array([])
                continue
            # build rolling windows matrix
            shape = (nwin - sync_window + 1, sync_window)
            strides = (A_s.strides[0], A_s.strides[0])
            A_windows = np.lib.stride_tricks.as_strided(A_s, shape=shape, strides=strides)
            B_windows = np.lib.stride_tricks.as_strided(B_s, shape=shape, strides=strides)
            # compute correlation per row
            # vectorized: subtract means, compute dot/div
            A_mean = A_windows.mean(axis=1, keepdims=True)
            B_mean = B_windows.mean(axis=1, keepdims=True)
            A_d = A_windows - A_mean
            B_d = B_windows - B_mean
            num = np.sum(A_d * B_d, axis=1)
            den = np.sqrt(np.sum(A_d**2, axis=1) * np.sum(B_d**2, axis=1))
            r = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            results[f] = r
        return results
