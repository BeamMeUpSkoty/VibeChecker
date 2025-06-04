import parselmouth
import librosa
import numpy as np
from typing import Dict, List, Optional


class AudioFeatures:
    """
    Extract acoustic features from either:
      a WAV file on disk (via `path`), OR
      a raw NumPy array + sampling rate (via `array` & `sr`).

    Use .extract(features_list, verbose) to get only the features you want.
    """

    def __init__(
            self,
            path: Optional[str] = None,
            array: Optional[np.ndarray] = None,
            sr: Optional[int] = None,
    ):
        if array is not None and sr is not None:
            # Build a Parselmouth Sound from raw array
            # Parselmouth expects shape (n_samples,) or (n_channels, n_samples)
            if array.ndim == 1:
                self.sound = parselmouth.Sound(array, sampling_frequency=sr)
            else:
                # transpose so shape = (n_channels, n_samples)
                self.sound = parselmouth.Sound(array.T, sampling_frequency=sr)
            self._array = array
            self._sr = sr
            self.path = None
        elif path is not None:
            self.path = path
            self.sound = parselmouth.Sound(self.path)
            self._array = None
            self._sr = None
        else:
            raise ValueError("Either `path` or (`array`, `sr`) must be provided.")

    # ─────────────────────────────────────────────────────────
    # Internal helper: compute all F0 statistics (using librosa.pyin)
    # returns a dict with keys 'min_f0','max_f0','mean_f0','median_f0','sd_f0','f0_range','80_percentile_range'
    # ─────────────────────────────────────────────────────────
    def _compute_f0_stats(self, verbose: bool = False) -> Dict[str, float]:
        # 1) load or reuse raw audio for librosa
        if self._array is not None and self._sr is not None:
            y = self._array.flatten()
            sr = self._sr
        elif self.path is not None:
            y, sr = librosa.load(self.path, sr=None)
        else:
            return {}

        # 2) extract f0 contour via pyin
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        if f0 is None or not voiced_flag.any():
            return {}

        f0_voiced = f0[voiced_flag]
        min_f0 = float(np.min(f0_voiced))
        max_f0 = float(np.max(f0_voiced))
        mean_f0 = float(np.mean(f0_voiced))
        median_f0 = float(np.median(f0_voiced))
        sd_f0 = float(np.std(f0_voiced))
        f0_range = float(max_f0 - min_f0)

        # 80th percentile range
        perc80 = np.percentile(f0_voiced, 80)
        above80 = f0_voiced[f0_voiced > perc80]
        range80 = float(np.max(above80) - np.min(above80)) if len(above80) > 0 else 0.0

        stats = {
            "min_f0": min_f0,
            "max_f0": max_f0,
            "mean_f0": mean_f0,
            "median_f0": median_f0,
            "sd_f0": sd_f0,
            "f0_range": f0_range,
            "80_percentile_range": range80,
        }
        if verbose:
            print("F0 stats:")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f} Hz")
        return stats

    # ─────────────────────────────────────────────────────────
    # Internal helper: compute average pitch via Parselmouth
    # returns {'average_pitch': float}
    # ─────────────────────────────────────────────────────────
    def _compute_average_pitch(self, verbose: bool = False) -> Dict[str, float]:
        pitch = self.sound.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_voiced = pitch_values[pitch_values > 0]
        avg = float(np.mean(pitch_voiced)) if len(pitch_voiced) else 0.0
        if verbose:
            print(f"Average pitch: {avg:.2f} Hz")
        return {"average_pitch": avg}

    # ─────────────────────────────────────────────────────────
    # Internal helper: compute articulation rate (syllables/sec) via TextGrid in Parselmouth
    # expects intervals labeled "word"
    # returns {'syllables_per_second': float}
    # ─────────────────────────────────────────────────────────
    def _compute_articulation_rate(self, verbose: bool = False) -> Dict[str, float]:
        tg = self.sound.to_textgrid()
        total_duration = self.sound.get_total_duration()
        syllable_count = 0
        pause_dur = 0.0

        for i in range(1, tg.get_number_of_intervals() + 1):
            interval = tg.get_interval(i)
            if interval[2] != "word":
                continue
            start_time, end_time, label = interval[0], interval[1], interval[3]
            duration = end_time - start_time
            if duration <= 0:
                continue
            if pause_dur > 0:
                total_duration -= pause_dur
                pause_dur = 0.0
            num_syl = len(label.split("-"))
            syllable_count += num_syl
            total_duration -= duration

            if i + 1 <= tg.get_number_of_intervals():
                next_int = tg.get_interval(i + 1)
                gap = next_int[0] - end_time
                if gap > 0:
                    pause_dur += gap

        rate = syllable_count / total_duration if total_duration > 0 else 0.0
        if verbose:
            print(f"Syllables per second: {rate:.2f}")
        return {"syllables_per_second": rate}

    # ─────────────────────────────────────────────────────────
    # Internal helper: compute cumulative speech index (CSI) from pitch contour
    # returns {'csi': float}
    # ─────────────────────────────────────────────────────────
    def _compute_csi(self, verbose: bool = False) -> Dict[str, float]:
        pitch = self.sound.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        voiced = pitch_values[pitch_values > 0]
        time_step = pitch.get_time_step()
        csi_val = 0.0
        for i in range(1, len(voiced)):
            slope = (voiced[i] - voiced[i - 1]) / time_step
            csi_val += abs(slope)
        if verbose:
            print(f"Cumulative Speech Index: {csi_val:.2f}")
        return {"csi": csi_val}

    # ─────────────────────────────────────────────────────────
    # Internal helper: compute intensity stats via Parselmouth
    # returns {'min_intensity','max_intensity','mean_intensity','sd_intensity'}
    # ─────────────────────────────────────────────────────────
    def _compute_intensity_stats(self, verbose: bool = False) -> Dict[str, float]:
        intensity = self.sound.to_intensity()
        min_i = parselmouth.praat.call(intensity, "Get minimum", 0.0, 0.0, "Parabolic")
        max_i = parselmouth.praat.call(intensity, "Get maximum", 0.0, 0.0, "Parabolic")
        mean_i = parselmouth.praat.call(intensity, "Get mean", 0.0, 0.0, "dB")
        sd_i = parselmouth.praat.call(intensity, "Get standard deviation", 0.0, 0.0)
        stats = {
            "min_intensity": float(min_i),
            "max_intensity": float(max_i),
            "mean_intensity": float(mean_i),
            "sd_intensity": float(sd_i),
        }
        if verbose:
            print("Intensity stats:")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f} dB")
        return stats

    # ─────────────────────────────────────────────────────────
    # Public: extract only the requested features
    # features: a list of strings (must be among the supported names)
    # ─────────────────────────────────────────────────────────
    def extract(
            self, features: List[str], verbose: bool = False
    ) -> Dict[str, float]:
        """
        Returns a dict mapping each requested feature → its numeric value.
        Supported feature keys:
          - F0 stats:      "min_f0", "max_f0", "mean_f0", "median_f0", "sd_f0", "f0_range", "80_percentile_range"
          - Avg pitch:     "average_pitch"
          - Articulation:  "syllables_per_second"
          - CSI:           "csi"
          - Intensity:     "min_intensity", "max_intensity", "mean_intensity", "sd_intensity"
        """
        out: Dict[str, float] = {}
        # 1) If any f0‐related keys requested, compute all f0 stats once
        f0_keys = {
            "min_f0",
            "max_f0",
            "mean_f0",
            "median_f0",
            "sd_f0",
            "f0_range",
            "80_percentile_range",
        }
        if any(k in features for k in f0_keys):
            f0_stats = self._compute_f0_stats(verbose)
            for k in features:
                if k in f0_stats:
                    out[k] = f0_stats[k]

        # 2) If "average_pitch" requested
        if "average_pitch" in features:
            out.update(self._compute_average_pitch(verbose))

        # 3) If any intensity key requested:
        intensity_keys = {"min_intensity", "max_intensity", "mean_intensity", "sd_intensity"}
        if any(k in features for k in intensity_keys):
            int_stats = self._compute_intensity_stats(verbose)
            for k in features:
                if k in int_stats:
                    out[k] = int_stats[k]

        # 4) If "syllables_per_second" requested
        if "syllables_per_second" in features:
            out.update(self._compute_articulation_rate(verbose))

        # 5) If "csi" requested
        if "csi" in features:
            out.update(self._compute_csi(verbose))

        # 6) Any unrecognized keys?
        for k in features:
            if k not in out:
                out[k] = 0.0  # or raise an error/warning if you prefer
        return out
