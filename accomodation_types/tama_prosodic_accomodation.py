from audio_features.audio_features import AudioFeatures
from accomodation_types.base_accomodation import BaseAccommodation
import numpy as np
import matplotlib.pyplot as plt


class TAMAProsodicAcommodation(BaseAccommodation):
    """ Time Aligned Moving Average (TAMA) analyses the audio in a
    fixed window, averageing values over the duration of the window.
    This method is based off extracting average prosodic values by for
    each speaker from a series of overlapping fixed length windows

    For each fixed window [t₀, t₀ + window_len), we:
      - Gather all speaker‐A utterance‐segments that overlap [t₀, t₀+window_len),
        concatenate their audio, and compute f0/intensity/articulation_rate on that chunk.
      - Do the same for speaker‐B.
    Convergence = PearsonCorr( |A[t] – B[t]| , t ).
    Synchrony  = sliding‐window PearsonCorr over the two feature‐time‐series.

    de Looze 2011; Fixed window = 20s, Time step = 10s, weighed average

    """

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
        :param audio_path: path to a single WAV (mixed) containing both speakers.
        :param transcript_csv: path to CSV with columns start,end,text,speaker (exactly 2 distinct speakers).
        :param requested_features: list of feature names from AudioFeatures.extract()
        :param window_len: window length in seconds (e.g. 20.0).
        :param hop: hop size in seconds (e.g. 10.0).
        """

        super().__init__(audio_path, transcript_csv, requested_features=requested_features, verbose=verbose)

        self.window_len = window_len
        self.hop = hop

        # We already have self.duration from Base; compute all window start times
        self.window_starts = np.arange(
            0.0, self.duration - self.window_len + 1e-8, self.hop
        )

        # name the two speaker IDs
        self.speaker_A = self.speaker_ids[0]
        self.speaker_B = self.speaker_ids[1]

    def get_accommodation(self) -> dict[str, np.ndarray]:
        """
        For each window index i (i = 0..nwin−1):
          - t0 = window_starts[i], t1 = t0 + window_len
          - chunk_A = all speaker_A audio in [t0,t1)
          - chunk_B = all speaker_B audio in [t0,t1)
          - feats_A = _wrap_and_extract(chunk_A)   # dict mapping feature→value
          - feats_B = _wrap_and_extract(chunk_B)

        Returns a dict:
          {
            feature1: np.ndarray of shape (nwin, 2),
            feature2: np.ndarray of shape (nwin, 2),
            ...
          }
        where [:,0] = speaker A’s value, [:,1] = speaker B’s value, length = nwin.
        """
        nwin = len(self.window_starts)
        nfeat = len(self.requested_features)

        # Initialize a dictionary of zero arrays, one per feature
        accom: dict[str, np.ndarray] = {
            feat: np.zeros((nwin, 2), dtype=float) for feat in self.requested_features
        }

        for idx, t0 in enumerate(self.window_starts):
            t1 = t0 + self.window_len

            # 1) gather raw audio chunks
            chunk_A = self._get_speaker_window_chunk(self.speaker_A, t0, t1)
            chunk_B = self._get_speaker_window_chunk(self.speaker_B, t0, t1)

            # 2) wrap & extract requested features
            feats_A = self._wrap_and_extract(chunk_A)
            feats_B = self._wrap_and_extract(chunk_B)

            # 3) fill arrays
            for f in self.requested_features:
                accom[f][idx, 0] = feats_A.get(f, 0.0)
                accom[f][idx, 1] = feats_B.get(f, 0.0)

        return accom

    def get_convergence(self) -> dict[str, float]:
        """
        For each requested feature f:
          - A_series = accom[f][:,0], B_series = accom[f][:,1]
          - d = |A_series − B_series|, t = [0..nwin−1]
          - r = PearsonCorr(d, t)
        Returns { f: r, … }.
        """
        accom = self.get_accommodation()
        results: dict[str, float] = {}
        for f in self.requested_features:
            pairs = accom[f]  # shape (nwin, 2)
            d = np.abs(pairs[:, 0] - pairs[:, 1])
            t = np.arange(len(d))
            results[f] = self._pearsonr(d, t)
        return results

    def get_synchrony(self, sync_window: int = 5) -> dict[str, np.ndarray]:
        """
        Sliding‐window Pearson over each feature‐stream.
        For each feature f:
          - A_series = accom[f][:,0], B_series = accom[f][:,1]
          - r_i = PearsonCorr(A[i:i+sync_window], B[i:i+sync_window])
            for i=0..(nwin‐sync_window)
        Returns { f: np.ndarray(length = nwin‐sync_window+1), … }.
        """
        accom = self.get_accommodation()
        nwin = accom[self.requested_features[0]].shape[0]
        results: dict[str, np.ndarray] = {}

        for f in self.requested_features:
            arr = accom[f]
            A_s = arr[:, 0]
            B_s = arr[:, 1]
            rs = []
            for i in range(nwin - sync_window + 1):
                segA = A_s[i: i + sync_window]
                segB = B_s[i: i + sync_window]
                rs.append(self._pearsonr(segA, segB))
            results[f] = np.array(rs)
        return results

    def get_visualization(self, output_path: str = None):
        """
        Plot each requested feature’s trajectories and distances. Then print r_convergence and
        mean(r_synchrony). If output_path is given, save the figure there.
        """

        accom = self.get_accommodation()
        conv = self.get_convergence()
        sync = self.get_synchrony()
        nwin = accom[self.requested_features[0]].shape[0]
        t = self.window_starts
        nf = len(self.requested_features)

        fig, axes = plt.subplots(nf, 2, figsize=(10, 4 * nf))
        if nf == 1:
            axes = np.array([[axes[0], axes[1]]])  # ensure 2D indexing

        for row, f in enumerate(self.requested_features):
            A_vals = accom[f][:, 0]
            B_vals = accom[f][:, 1]
            dist = np.abs(A_vals - B_vals)

            ax1 = axes[row, 0]
            ax1.plot(t, A_vals, "-o", label=f"A_{f}")
            ax1.plot(t, B_vals, "-s", label=f"B_{f}")
            ax1.set_title(f"{f} trajectories (TAMA)")
            ax1.legend()

            ax2 = axes[row, 1]
            ax2.plot(t, dist, "-x", color="gray", label="|A−B|")
            ax2.set_title(f"{f} distance per window")
            ax2.legend()

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path)
        else:
            plt.show()

        print("\n=== TAMA Accommodation Summary ===")
        for f in self.requested_features:
            mean_sync = float(sync[f].mean()) if sync[f].size > 0 else 0.0
            print(f"{f}: r_convergence = {conv[f]:.4f}, mean_r_synchrony = {mean_sync:.4f}")



