from abc import ABC, abstractmethod
import csv
import numpy as np
import soundfile as sf
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from audio_features.audio_features import AudioFeatures

class BaseAccommodation(ABC):
    """
    Abstract base class for prosodic accommodation.

    Requires:
      - one WAV file (mono or stereo)
      - one CSV transcript with columns: start,end,text,speaker
        (where 'speaker' must take exactly two distinct values).

    After initialization:
      self.audio        → numpy array of shape (n_samples,) or (n_samples, n_channels)
      self.sr           → sampling rate (int)
      self.speaker_ids  → sorted list of the two unique speaker IDs (e.g. ['A','B'])
      self.utts_by_speaker →
            {
              speaker_id_1: [ {'start':..., 'end':..., 'text':...}, … ],
              speaker_id_2: [ {'start':..., 'end':..., 'text':...}, … ],
            }
      self.duration     → total duration of the loaded audio (in seconds)

      self.requested_features (e.g. ['mean_f0','sd_f0','mean_intensity'
    """

    def __init__(
        self,
        audio_path: str,
        transcript_csv: str,
        requested_features: List[str] = None,
        frame_duration: float = 10.0,
        verbose: bool = False,
    ):
        """
        :param audio_path:        path to mixed‐speaker WAV.
        :param transcript_csv:    path to CSV with columns [start,end,text,speaker].
        :param requested_features:
                   list of feature‐names to extract (must match keys from AudioFeatures.extract()).
                   If None, defaults to ['mean_f0','mean_intensity','syllables_per_second'].
        : param frame_duration:
        :param verbose:           pass to feature extractor if you want prints.
        """

        self.audio_path = audio_path
        self.transcript_csv = transcript_csv
        self.frame_duration = frame_duration
        self.verbose = verbose

        # Load entire audio
        self.audio, self.sr = sf.read(audio_path, dtype="float32")
        self.duration = self.audio.shape[0] / self.sr

        # Parse CSV transcript into a list of dicts
        all_utts = []
        with open(transcript_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = float(row["start"])
                    end = float(row["end"])
                    speaker = row["speaker"]
                    text = row.get("text", "")
                except (KeyError, ValueError):
                    raise ValueError(
                        "Each row in transcript CSV must have 'start','end','speaker'."
                    )
                if not (0 <= start < end <= self.duration):
                    raise ValueError(
                        f"Invalid times: start={start}, end={end}, audio duration={self.duration:.2f}s."
                    )
                all_utts.append({"start": start, "end": end, "text": text, "speaker": speaker})

        if not all_utts:
            raise ValueError("Transcript CSV is empty or malformed.")

        # Identify exactly two speaker IDs
        speakers = sorted({utt["speaker"] for utt in all_utts})
        if len(speakers) != 2:
            raise ValueError(
                f"Transcript CSV must have exactly two speaker labels; found {len(speakers)}: {speakers}"
            )
        self.speaker_ids = speakers  # e.g. ['A', 'B']
        self.speaker_A, self.speaker_B = self.speaker_ids

        # Split utterances by speaker
        self.utts_by_speaker = {
            spk: sorted([u for u in all_utts if u["speaker"] == spk], key=lambda u: u["start"])
            for spk in self.speaker_ids
        }

        # Store requested_features (or default)
        if requested_features is None:
            # default: mean and SD f0 + mean intensity + speech rate
            requested_features = ["mean_f0", "sd_f0", "mean_intensity", "syllables_per_second"]
        self.requested_features = requested_features

    @abstractmethod
    def get_accommodation(self) -> Dict[str, np.ndarray]:
        """
        Compute per‐time‐step accommodation values for each feature.

        Subclasses should:
          - Decide on a “time‐base” (e.g. turn index, window index).
          - For each time‐step t, gather A_chunk and B_chunk: _get_speaker_window_chunk(...).
          - Call _wrap_and_extract(...) on each chunk to get a dict of feature values.
          - Store feature values in two arrays of shape (T, len(requested_features)).

        Returns a dict mapping each feature name -> an array of shape (T, 2),
        where [:,0] = speaker A’s value, [:,1] = speaker B’s value, for T time‐steps.
            {
                'f0': np.ndarray(shape=(T,2)), -> (speakerA_value, speakerB_value) per time‐step
                'intensity': np.ndarray(shape=(T,2)),
                'articulation_rate': np.ndarray(shape=(T,2))
             }

        The definition of “time‐step” (turn‐exchange index, fixed window, etc.) is left to subclasses.
        """
        pass

    @abstractmethod
    def get_convergence(self) -> Dict[str, float]:
        """ Computes a global convergence statistic per feature.

        For each feature f in self.requested_features:
          - Let A_t, B_t be the time‐series of that feature for t=0..T−1.
          - Compute r_f = PearsonCorr(d, t). Return {f: r_f, ...}.
          - Either compute one “turn‐taking” synchrony (Pearson(A[:-1], B[1:])),
            or a sliding‐window Pearson. Return {f: np.array([...]), ...}.

        For each feature, define a distance series d_t = |A_t – B_t| over time‐steps t=0..T−1,
        and return PearsonCorr(d, t). A negative correlation indicates convergence.

        Returns:
          {
            'f0': float,
            'intensity': float,
            'articulation_rate': float
          }
        """
        pass

    def _pearsonr(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Pearson’s r between 1D arrays x and y.  If denominator is zero, returns 0.0.
        """
        if len(x) != len(y):
            raise ValueError("Inputs to _pearsonr must have the same length.")
        mx = np.mean(x)
        my = np.mean(y)
        num = np.sum((x - mx) * (y - my))
        den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
        return float(num / den) if den != 0 else 0.0

    @staticmethod
    def _distance(self, a: float, b: float) -> float:
        """Absolute difference between two scalar feature values."""
        return abs(a - b)

    def _load_audio_segment(self, start: float, end: float) -> np.ndarray:
        """
        Given start/end in seconds, return the corresponding slice of the loaded audio array.
        If audio is multi‐channel, returns all channels for that time segment.
        """
        start_idx = int(start * self.sr)
        end_idx = int(end * self.sr)
        return self.audio[start_idx:end_idx]

    def _get_speaker_window_chunk(self, speaker: str, t0: float, t1: float) -> np.ndarray:
        """
        Return a 1D np.ndarray of all audio samples (concatenated) for `speaker`
        that overlap the interval [t0, t1).  We look at each utterance in
        self.utts_by_speaker[speaker], clip it to [t0, t1), slice from self.audio,
        and concatenate them in chronological order.  If no overlap, return np.array([]).
        """
        chunks = []
        for utt in self.utts_by_speaker[speaker]:
            utt_start = utt['start']
            utt_end = utt['end']
            # If no overlap, skip
            if utt_end <= t0 or utt_start >= t1:
                continue
            # Compute the overlap segment
            seg_start = max(utt_start, t0)
            seg_end = min(utt_end, t1)
            if seg_end <= seg_start:
                continue
            audio_seg = self._load_audio_segment(seg_start, seg_end)
            if audio_seg.size > 0:
                chunks.append(audio_seg)
        if not chunks:
            return np.array([], dtype=self.audio.dtype)
        # Concatenate along time‐axis (1D or 2D if stereo)
        return np.concatenate(chunks, axis=0)

    def _wrap_and_extract(self, array_chunk: np.ndarray) -> Dict[str, float]:
        """
        :param array_chunk: np.ndarray of shape (n_samples,) or (n_samples,n_channels).
        :returns: dict mapping each feature in self.requested_features -> its computed value.
        """
        if array_chunk.size == 0:
            # if empty chunk, return 0.0 for all requested_features
            return {f: 0.0 for f in self.requested_features}

        af = AudioFeatures(array=array_chunk, sr=self.sr)
        return af.extract(self.requested_features, verbose=self.verbose)

    def dynamic_synchrony(self, A_vals: np.ndarray, B_vals: np.ndarray, window: int = 10, hop: int = 5):
        """
        Slide a window of `window` frames, hop by `hop`, compute Pearson-r.

        Returns:
          rs    : np.ndarray of correlation values
          times : np.ndarray of midpoint times in seconds
        """
        n = len(A_vals)
        rs, times = [], []
        for start in range(0, n - window + 1, hop):
            segA = A_vals[start:start+window]
            segB = B_vals[start:start+window]
            r, _ = pearsonr(segA, segB)
            rs.append(r)
            mid = (start + window / 2) * self.frame_duration
            times.append(mid)
        return np.array(rs), np.array(times)

    def phase_metrics(self, rs: np.ndarray, window: int = 10) -> Dict[str, float]:
        """
        Count & sum time in phases based on rs thresholds:
          - synchrony   : r >=  0.5
          - asynchrony : r <= -0.5
          - maintenance: otherwise
        """
        sync_mask  = rs >=  0.5
        async_mask = rs <= -0.5
        maint_mask = ~(sync_mask | async_mask)
        n_sync, n_async, n_maint = sync_mask.sum(), async_mask.sum(), maint_mask.sum()
        t_sync  = n_sync  * window * self.frame_duration
        t_async = n_async * window * self.frame_duration
        t_maint = n_maint * window * self.frame_duration
        return {
            'n_sync': n_sync,   'time_sync':  t_sync,
            'n_async': n_async, 'time_async': t_async,
            'n_maint': n_maint, 'time_maint': t_maint,
        }

    def get_synchrony_features(self) -> Dict[str, Dict]:
        """
        Compute dynamic-synchrony metrics and phases for each requested feature.
        Returns a dict mapping feature -> stats dict including r_values and r_times.
        """
        accom = self.get_accommodation()
        out = {}
        for f in self.requested_features:
            A, B = accom[f][:,0], accom[f][:,1]
            rs, times = self.dynamic_synchrony(A, B)
            stats = self.phase_metrics(rs)
            out[f] = {**stats, 'r_values': rs, 'r_times': times}
        return out

    def get_state_features(self,
                           window: int = 10,
                           hop: int = 5,
                           thresh: float = 0.5
                          ) -> Dict[str, np.ndarray]:
        """
        For each requested feature, compute the sequence of 1–7 “states”
        across the interaction.
        Returns a dict: feature → array of state‐IDs.
        """
        accom = self.get_accommodation()
        out = {}
        for f in self.requested_features:
            A = accom[f][:,0]
            B = accom[f][:,1]
            out[f] = self.dynamic_states(A, B,
                                        window=window,
                                        hop=hop,
                                        thresh=thresh)
        return out

    def dynamic_states(self,
                       A_vals: np.ndarray,
                       B_vals: np.ndarray,
                       window: int = 10,
                       hop: int = 5,
                       thresh: float = 0.5
                      ) -> np.ndarray:
        """
        Slide (window×hop) over A_vals, B_vals and return for each window
        an integer 1–7 corresponding to the De Looze “state”:
          1=maintenance,
          2=synchrony,
          3=convergence,
          4=synchrony+convergence,
          5=asynchrony,
          6=divergence,
          7=asynchrony+divergence
        """
        n = len(A_vals)
        d = np.abs(A_vals - B_vals)
        states = []
        for start in range(0, n - window + 1, hop):
            segA = A_vals[start:start+window]
            segB = B_vals[start:start+window]
            segD = d[start:start+window]
            # synchrony / asynchrony
            r_sync, _ = pearsonr(segA, segB)
            is_sync  = r_sync >=  thresh
            is_async = r_sync <= -thresh
            # convergence / divergence
            idxs = np.arange(window)
            r_conv, _ = pearsonr(segD, idxs)
            is_conv  = r_conv <= -thresh
            is_div   = r_conv >=  thresh

            # map to state ID
            if   not any((is_sync, is_async, is_conv, is_div)):
                state = 1
            elif  is_sync  and  is_conv:
                state = 4
            elif  is_sync:
                state = 2
            elif  is_conv:
                state = 3
            elif  is_async and  is_div:
                state = 7
            elif  is_async:
                state = 5
            elif  is_div:
                state = 6
            else:
                state = 1
            states.append(state)
        return np.array(states)

    def _plot_trajectories(
        self,
        ax: Axes,
        feature: str,
        accom: Dict[str, np.ndarray],
        t_index: np.ndarray,
        loess_frac: Optional[float]
    ) -> None:
        A = accom[feature][:, 0]
        B = accom[feature][:, 1]
        ax.plot(t_index, A, '-o', alpha=0.6, label=f"{self.speaker_A}")
        ax.plot(t_index, B, '-s', alpha=0.6, label=f"{self.speaker_B}")
        ax.set(title=f"{feature} trajectories", xlabel="Turn #", ylabel=feature)
        if loess_frac is not None:
            lo_A = lowess(endog=A, exog=t_index, frac=loess_frac, return_sorted=True)
            lo_B = lowess(endog=B, exog=t_index, frac=loess_frac, return_sorted=True)
            ax.plot(lo_A[:, 0], lo_A[:, 1], '--', label=f"{self.speaker_A} LOESS")
            ax.plot(lo_B[:, 0], lo_B[:, 1], '--', label=f"{self.speaker_B} LOESS")
        ax.legend()

    def _plot_distance(
        self,
        ax: Axes,
        feature: str,
        accom: Dict[str, np.ndarray],
        t_index: np.ndarray
    ) -> None:
        dist = np.abs(accom[feature][:, 0] - accom[feature][:, 1])
        ax.plot(t_index, dist, '-x', color='gray', alpha=0.6)
        ax.set(title=f"{feature} |A–B|", xlabel="Turn #", ylabel="Distance")

    def _plot_synchrony(
        self,
        ax: Axes,
        feature: str,
        sync_feats: Dict[str, Dict[str, np.ndarray]],
        mode: str,
        window: int,
        thresh: float,
        frame_duration: float
    ) -> None:
        info = sync_feats[feature]
        times = info["r_times"]
        rs = info["r_values"]

        if mode == "combined":
            ax.plot(times, rs, '-o', label="dynamic r")
            static_r = info["_static"]
            ax.axhline(static_r, color='red', linestyle='--', label="turn r")
        else:
            ax.plot(times, rs, '-o', label=f"{mode} r")

        half_span = (window * frame_duration) / 2
        pos_intervals = [(t - half_span, t + half_span) for r, t in zip(rs, times) if r >= thresh]
        neg_intervals = [(t - half_span, t + half_span) for r, t in zip(rs, times) if r <= -thresh]

        def _merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            merged: List[Tuple[float, float]] = []
            for start, end in sorted(intervals):
                if not merged or start > merged[-1][1]:
                    merged.append((start, end))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            return merged

        for start, end in _merge(pos_intervals):
            ax.axvspan(start, end, color='red', alpha=0.3)
        for start, end in _merge(neg_intervals):
            ax.axvspan(start, end, color='green', alpha=0.3)

        ax.axhline(thresh, linestyle='--', color='red')
        ax.axhline(-thresh, linestyle='--', color='green')
        ax.set(
            title=f"{feature} synchrony ({mode})",
            xlabel="Time (s)",
            ylabel="r"
        )
        ax.legend(loc='upper right')

    def _plot_transition_matrix(
        self,
        ax: Axes,
        feature: str,
        state_feats: Dict[str, np.ndarray]
    ) -> None:
        states = state_feats[feature]
        max_state = states.max()
        counts = np.zeros((max_state, max_state), dtype=int)
        for prev, nxt in zip(states[:-1], states[1:]):
            counts[prev - 1, nxt - 1] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / np.where(row_sums == 0, 1, row_sums)

        im = ax.imshow(probs, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"{feature} state transitions")
        ax.set_xlabel("Next state")
        ax.set_ylabel("Previous state")
        ax.set_xticks(range(max_state))
        ax.set_yticks(range(max_state))

        for i in range(max_state):
            for j in range(max_state):
                color = "white" if probs[i, j] > 0.5 else "black"
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", color=color)

        plt.colorbar(im, ax=ax, label="P(next|prev)")

    def _plot_raster(
        self,
        ax: Axes,
        state_series: Dict[str, np.ndarray],
        window: int,
        frame_duration: float
    ) -> None:
        """
        state_series: mapping from label to its state-array.
        """
        for row, (label, states) in enumerate(state_series.items()):
            times = (np.arange(len(states)) + window / 2) * frame_duration
            ax.scatter(times, [row] * len(states), c=states, cmap='tab10', s=10)
        ax.set_yticks(range(len(state_series)))
        ax.set_yticklabels(list(state_series.keys()))
        ax.set_xlabel("Time (s)")
        ax.set_title("State raster across sequences")

    def _plot_gantt(
            self,
            ax: Axes,
            feature: str,
            state_feats: Dict[str, np.ndarray],
            window: int,
            hop: int,
            frame_duration: float
    ) -> None:
        """
        On ax, draw a Gantt‐style timeline for `feature`’s state sequence.
        Each state i is represented as a block of duration hop*frame_duration,
        starting at time i*hop*frame_duration; contiguous runs of the same
        state get merged into a single longer block.
        """
        states = state_feats[feature]  # 1D array of state IDs
        n = len(states)
        # compute non-overlapping start times and durations
        starts = np.arange(n) * hop * frame_duration
        lengths = np.full(n, hop * frame_duration)
        # merge contiguous runs
        segments = []
        cur_state = states[0]
        cur_start = starts[0]
        cur_len = lengths[0]
        for s, dur, st in zip(starts[1:], lengths[1:], states[1:]):
            if st == cur_state:
                cur_len += dur
            else:
                segments.append((cur_start, cur_len, cur_state))
                cur_state, cur_start, cur_len = st, s, dur
        segments.append((cur_start, cur_len, cur_state))

        # pick colors from tab10 (state IDs start at 1)
        cmap = plt.get_cmap('tab10')
        for start, dur, st in segments:
            ax.broken_barh(
                [(start, dur)],
                (0, 1),
                facecolors=cmap((st - 1) % 10),
                edgecolor='black',
                alpha=0.8
            )

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{feature} state timeline (Gantt)")
        # build a custom legend
        unique_states = sorted(set(states))
        handles = [
            plt.Line2D([0], [0], color=cmap((st - 1) % 10), lw=8, label=f"State {st}")
            for st in unique_states
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    def _shade_states_background(
        self,
        ax: Axes,
        feature: str,
        state_feats: Dict[str, np.ndarray],
        window: int,
        frame_duration: float,
        alpha: float = 0.15
    ) -> None:
        """
        Shades the background of `ax` in bands according to the state sequence
        for `feature`. Each band spans one state-window (window*frame_duration).
        """
        states = state_feats[feature]            # e.g. array([1,2,2,3,...])
        n = len(states)
        # compute center times of each state-window
        centers = (np.arange(n) + window/2) * frame_duration
        half = (window * frame_duration) / 2

        cmap = plt.get_cmap("tab10")
        for st, ct in zip(states, centers):
            start, end = ct - half, ct + half
            ax.axvspan(start, end,
                       facecolor=cmap((st-1) % 10),
                       alpha=alpha,
                       edgecolor="none")

    def get_visualization(
        self,
        output_path: Optional[str] = None,
        loess_frac: Optional[float] = None,
        window: int = 10,
        hop: int = 5,
        thresh: float = 0.5
    ) -> None:
        accom = self.get_accommodation()
        n_steps = next(iter(accom.values())).shape[0]
        t_index = np.arange(n_steps)
        full_times = t_index * self.frame_duration
        mode = getattr(self, "synchrony_mode", "dynamic")

        dynamic_feats = self.get_synchrony_features()
        state_feats = self.get_state_features(window, hop, thresh)

        # build sync_feats (turn / combined / dynamic)
        if mode == "turn":
            static = self.get_synchrony()
            sync_feats = {
                f: {
                    "r_values": np.full_like(full_times, static[f], dtype=float),
                    "r_times": full_times
                }
                for f in self.requested_features
            }
        elif mode == "combined":
            static = self.get_synchrony()
            sync_feats = {}
            for f in self.requested_features:
                sync_feats[f] = {
                    "r_values": dynamic_feats[f]["r_values"],
                    "r_times": dynamic_feats[f]["r_times"],
                    "_static": static[f]
                }
        else:
            sync_feats = dynamic_feats

        nf = len(self.requested_features)
        fig, axes = plt.subplots(nf, 3, figsize=(20, 3 * nf))
        if nf == 1:
            axes = np.array([axes])

        for row, f in enumerate(self.requested_features):
            ax0, ax1, ax2 = axes[row]
            self._plot_trajectories(ax0, f, accom, t_index, loess_frac)
            self._plot_distance(ax1, f, accom, t_index)
            self._plot_synchrony(ax2, f, sync_feats, mode, window, thresh, self.frame_duration)
            # Choose either transition matrix or raster here:
            #self._plot_transition_matrix(ax3, f, state_feats)
            #self._plot_raster(ax3, state_feats, window, self.frame_duration)
            #self._plot_gantt(ax3, f, state_feats, window, hop, self.frame_duration)
            # after plotting A and B on ax0
            #self._shade_states_background(
            #    ax0, f, state_feats, window, self.frame_duration, alpha=0.1
            #)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path)
        else:
            plt.show()