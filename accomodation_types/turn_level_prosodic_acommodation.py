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

    def get_synchrony(self) -> dict[str, float]:
        """
        Turn-Taking synchrony: For each feature f:
          - Let A_prev = [A_0, A_1, …, A_{n-2}], B_curr = [B_1, …, B_{n-1}].
          - Return PearsonCorr(A_prev, B_curr). If n_exchanges ≤ 1, return 0.0.
        """
        accom = self.get_accommodation()
        results: Dict[str, float] = {}
        for f, pairs in accom.items():
            A_vals = pairs[:-1,0]
            B_vals = pairs[1:,1]
            if A_vals.size < 1 or np.std(A_vals)==0 or np.std(B_vals)==0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(A_vals, B_vals)[0,1])
            results[f] = corr
        return results

    '''
    def get_visualization(self, output_path: str = None, loess_frac: float = 0.3):
        """
        Plot each feature’s trajectories and distances across turn indices,
        then overlay LOESS‐smoothed curves and print r_convergence and r_synchrony.
        """
        accom = self.get_accommodation()
        conv = self.get_convergence()
        sync = self.get_synchrony()

        n_exchanges = accom[self.requested_features[0]].shape[0]
        t = np.arange(n_exchanges)
        nf = len(self.requested_features)

        fig, axes = plt.subplots(nf, 2, figsize=(10, 4 * nf))
        if nf == 1:
            axes = np.array([[axes[0], axes[1]]])  # ensure 2D

        for row, f in enumerate(self.requested_features):
            A_vals = accom[f][:, 0]
            B_vals = accom[f][:, 1]
            dist = np.abs(A_vals - B_vals)

            # —— Trajectories subplot —— #
            ax1 = axes[row, 0]
            ax1.plot(t, A_vals, "-o", label=f"{self.speaker_A}_{f}", alpha=0.6)
            ax1.plot(t, B_vals, "-s", label=f"{self.speaker_B}_{f}", alpha=0.6)

            # compute & plot LOESS for A and B
            lo_A = lowess(endog=A_vals, exog=t, frac=loess_frac, return_sorted=True)
            lo_B = lowess(endog=B_vals, exog=t, frac=loess_frac, return_sorted=True)
            ax1.plot(lo_A[:, 0], lo_A[:, 1],
                     linestyle="--", linewidth=2,
                     label=f"{self.speaker_A}_{f} (LOESS)")
            ax1.plot(lo_B[:, 0], lo_B[:, 1],
                     linestyle="--", linewidth=2,
                     label=f"{self.speaker_B}_{f} (LOESS)")

            ax1.set_title(f"{f} trajectories (Turn‐Taking)")
            ax1.set_xlabel("Turn Index")
            ax1.set_ylabel(f)
            ax1.legend()

            # —— Distance subplot —— #
            ax2 = axes[row, 1]
            ax2.plot(t, dist, "-x", color="gray", label="|A−B|", alpha=0.6)

            # compute & plot LOESS for distance
            lo_D = lowess(endog=dist, exog=t, frac=loess_frac, return_sorted=True)
            ax2.plot(lo_D[:, 0], lo_D[:, 1],
                     linestyle="--", linewidth=2,
                     label="|A−B| (LOESS)")

            ax2.set_title(f"{f} |A−B| per turn")
            ax2.set_xlabel("Turn Index")
            ax2.set_ylabel("Distance")
            ax2.legend()

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path)
        else:
            plt.show()

        print("\n=== Turn-Taking Accommodation Summary ===")
        for f in self.requested_features:
            r_conv = conv[f]
            r_sync = sync[f]
            print(f"{f}:   r_convergence = {r_conv:.4f},   r_synchrony = {r_sync:.4f}")
        '''