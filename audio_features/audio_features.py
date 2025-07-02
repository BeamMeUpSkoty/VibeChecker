import parselmouth
import numpy as np
import librosa
import argparse
from typing import Dict, List, Optional, Tuple

class AudioFeatures:
    """
      - Loads audio once and caches raw waveform
      - Pre-computes Pitch and Intensity for full audio
      - Acoustic-based articulation rate (no transcript needed)
      - Optional downsampling (default: True, target_sr=16000)
      - Caches per-segment statistics
      - Computes only requested features on demand
      - Robust fallback for missing voiced frames
    """
    def __init__(
        self,
        path: Optional[str] = None,
        array: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        downsample: bool = True,
        target_sr: int = 16000
    ) -> None:
        # Load audio
        if array is not None and sr is not None:
            y = array.flatten(); orig_sr = sr
        elif path:
            sound = parselmouth.Sound(path)
            y = sound.values.T.flatten(); orig_sr = sound.sampling_frequency
        else:
            raise ValueError("Provide either `path` or (`array` and `sr`).")

        # Downsample if needed
        if downsample and orig_sr != target_sr:
            from scipy.signal import resample_poly
            y = resample_poly(y, target_sr, orig_sr)
            sr = target_sr
        else:
            sr = orig_sr

        # Initialize audio and analysis objects
        self._waveform = y
        self._sr = sr
        self.sound = parselmouth.Sound(y, sampling_frequency=sr)
        self._pitch = self.sound.to_pitch()
        self._intensity = self.sound.to_intensity()

        # Cache: {(start,end,features): stats}
        self._seg_cache: Dict[Tuple[float, float, Tuple[str, ...]], Dict[str, float]] = {}

    def _compute_f0_stats(self, start: float, end: float) -> Dict[str, float]:
        times = self._pitch.xs()
        freqs = self._pitch.selected_array['frequency']
        mask = (times >= start) & (times <= end)
        voiced = freqs[mask & (freqs > 0)]
        if voiced.size == 0:
            return {k: 0.0 for k in ['min_f0','max_f0','mean_f0','median_f0','sd_f0','f0_range','80_percentile_range']}
        stats = {
            'min_f0': float(np.min(voiced)),
            'max_f0': float(np.max(voiced)),
            'mean_f0': float(np.mean(voiced)),
            'median_f0': float(np.median(voiced)),
            'sd_f0': float(np.std(voiced)),
            'f0_range': float(np.ptp(voiced))
        }
        perc80 = np.percentile(voiced, 80)
        high = voiced[voiced > perc80]
        stats['80_percentile_range'] = float(np.ptp(high)) if high.size else 0.0
        return stats

    def _compute_intensity_stats(self, start: float, end: float) -> Dict[str, float]:
        i_min = parselmouth.praat.call(self._intensity, 'Get minimum', start, end, 'Parabolic')
        i_max = parselmouth.praat.call(self._intensity, 'Get maximum', start, end, 'Parabolic')
        i_mean = parselmouth.praat.call(self._intensity, 'Get mean', start, end, 'dB')
        i_sd = parselmouth.praat.call(self._intensity, 'Get standard deviation', start, end)
        return {
            'min_intensity': float(i_min),
            'max_intensity': float(i_max),
            'mean_intensity': float(i_mean),
            'sd_intensity': float(i_sd)
        }

    def _compute_csi(self, start: float, end: float) -> Dict[str, float]:
        times = self._pitch.xs()
        freqs = self._pitch.selected_array['frequency']
        mask = (times >= start) & (times <= end)
        voiced = freqs[mask & (freqs > 0)]
        if voiced.size < 2:
            return {'csi': 0.0}
        tstep = self._pitch.get_time_step()
        csi_val = float(np.sum(np.abs(np.diff(voiced) / tstep)))
        return {'csi': csi_val}

    def _compute_acoustic_articulation_rate(
        self,
        start: float,
        end: float,
        hop_length: int = 512,
        backtrack: bool = False
    ) -> Dict[str, float]:
        """
        Estimate articulation rate (syllables/sec) via onset detection.
        Uses librosa.onset.onset_detect for robust peak finding.
        """
        # Extract segment waveform
        start_samp = int(start * self._sr)
        end_samp = int(end * self._sr)
        y_seg = self._waveform[start_samp:end_samp]
        if y_seg.size == 0:
            return {'articulation_rate': 0.0}

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y_seg,
            sr=self._sr,
            hop_length=hop_length,
            backtrack=backtrack,
            units='frames'
        )
        # Convert to time if needed:
        # onset_times = librosa.frames_to_time(onset_frames, sr=self._sr, hop_length=hop_length)
        count = len(onset_frames)

        duration = end - start
        rate = count / duration if duration > 0 else 0.0
        return {'articulation_rate': float(rate)}

    def _extract_segment(
        self,
        start: float,
        end: float,
        requested: List[str]
    ) -> Dict[str, float]:
        key = (start, end, tuple(sorted(requested)))
        if key in self._seg_cache:
            return self._seg_cache[key]

        stats: Dict[str, float] = {}
        f0_keys = {'min_f0','max_f0','mean_f0','median_f0','sd_f0','f0_range','80_percentile_range'}
        if any(k in requested for k in f0_keys): stats.update(self._compute_f0_stats(start, end))
        intensity_keys = {'min_intensity','max_intensity','mean_intensity','sd_intensity'}
        if any(k in requested for k in intensity_keys): stats.update(self._compute_intensity_stats(start, end))
        if 'csi' in requested: stats.update(self._compute_csi(start, end))
        if 'articulation_rate' in requested: stats.update(self._compute_acoustic_articulation_rate(start, end))

        out = {k: stats.get(k, 0.0) for k in requested}
        self._seg_cache[key] = out
        return out

    def extract(
        self,
        features: List[str],
        verbose: bool = False
    ) -> Dict[str, float]:
        duration = self.sound.get_total_duration()
        return self.extract_segment(features, 0.0, duration, verbose)

    def extract_segment(
        self,
        features: List[str],
        start: float,
        end: float,
        verbose: bool = False
    ) -> Dict[str, float]:
        result = self._extract_segment(start, end, features)
        if verbose:
            for k, v in result.items(): print(f"{k}: {v:.3f}")
        return result