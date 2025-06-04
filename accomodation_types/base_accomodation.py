from abc import ABC, abstractmethod
import csv
import numpy as np
import soundfile as sf
from typing import List, Dict

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
        verbose: bool = False,
    ):
        """
        :param audio_path:        path to mixed‐speaker WAV.
        :param transcript_csv:    path to CSV with columns [start,end,text,speaker].
        :param requested_features:
                   list of feature‐names to extract (must match keys from AudioFeatures.extract()).
                   If None, defaults to ['mean_f0','mean_intensity','syllables_per_second'].
        :param verbose:           pass to feature extractor if you want prints.
        """

        self.audio_path = audio_path
        self.transcript_csv = transcript_csv
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

    @abstractmethod
    def get_synchrony(self) -> dict:
        """
        Compute local/short‐term synchrony per feature.
        Typically, for each feature we have two parallel time‐series (A_t, B_t).  Then:
          – Either compute one global PearsonCorr(A[:-1], B[1:]) (turn‐taking style),
          – Or compute a sliding‐window PearsonCorr(A[t:t+W], B[t:t+W]) (TAMA/hybrid style).

        Returns:
          {
            'f0': np.ndarray(...),
            'intensity': np.ndarray(...),
            'articulation_rate': np.ndarray(...)
          }
        The length and interpretation of each array depend on the subclass’s chosen windowing.
        """
        pass

    @abstractmethod
    def get_visualization(self, output_path: str = None):
        """
        Produce diagnostic plots:
          - Raw feature trajectories for both speakers over time‐steps for each requested feature f.
          - Distance (|A_t−B_t|) vs. time‐step.
          - If sliding‐window, the synchrony curve vs. time‐step.

        If output_path is provided, save the figure there; otherwise, display on screen.

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
        TODO: See if this method changed.
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