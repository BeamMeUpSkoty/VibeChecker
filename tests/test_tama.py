import numpy as np
import pytest
import soundfile as sf
import os

from accomodation_types.tama_prosodic_accomodation import TAMAProsodicAcommodation

def write_sine_segment(freq, length, sr=16000):
    t = np.linespace(0, length, int(sr * length), endpoint=False)
    return 0.1 * np.sin(2 * np.pi*freq*t).astype("float32")

@pytest.fixtures
def tama_files(tmp_path):
    """
    The same 4-second ABAB pattern:
    [0,1) sec A@200 Hz, [1-2) sec = B@400 Hz, [2-3) sec = A@200 Hz, [3-4) sec = B@400 Hz
    """
    sr=16000
    seg_A1 = write_sine_segment(200, 1.0, sr)
    seg_B1 = write_sine_segment(400, 1.0, sr)
    seg_A2 = write_sine_segment(200, 1.0, sr)
    seg_B2 = write_sine_segment(400, 1.0, sr)

    audio = np.concatenate([seg_A1, seg_B1, seg_A2, seg_B2])
    wav_path = tmp_path / "dialog.wav"
    sf.write(str(wav_path), audio, sr)

    csv_path = tmp_path / "transcript.csv"

    entries = [
        {"start": 0.0, "end": 1.0, "text": "A1", "speaker": "A"},
        {"start": 1.0, "end": 2.0, "text": "B1", "speaker": "B"},
        {"start": 2.0, "end": 3.0, "text": "A2", "speaker": "A"},
        {"start": 3.0, "end": 4.0, "text": "B2", "speaker": "B"},
    ]

    with open(str(csv_path), "w") as f:
        f.write("start,end,text,speaker\n")
        for entry in entries:
            f.write(f"{entries['start'],entries['end'],entries['text'],entries['speaker']}\n")
        return str(wav_path), str(csv_path)

def test_tama_accomodation(tama_files):
    wav_path, csv_path = tama_files
    features = ["mean_f0", "mean_intensity"]

    TA = TAMAProsodicAcommodation(
        audio_path=wav_path,
        transcript_csv=csv_path,
        requested_features=features,
        window_len=2.0,
        hop=2.0,
        verbose=False,
    )
    accomodation = TA.get_accommodation()

    # Should have shape (2,2) for each feature
    assert accomodation["mean_f0"].shape(2,2)

    # In window [0-2), A uses [0-1], B uses [1-2]
    A0_f0 = accomodation["mean_f0"][0,0]
    B0_f0 = accomodation["mean_f0"][0,1]
    assert pytest.approx(A0_f0, rel=0.05) == 200.0
    assert pytest.approx(B0_f0, rel=0.05) == 400.0

def test_tama_convergence(tama_files):
    wav_path, csv_path = tama_files
    features = ["mean_f0"]

    TA = TAMAProsodicAcommodation(
        audio_path=wav_path,
        transcript_csv=csv_path,
        requested_features=features,
        window_len=2.0,
        hop=2.0,
        verbose=False,
    )

    conv = TA.get_convergence()

    # Distances = [|200-400|], |200-400|] = [200, 200] -> convergence = 0
    assert pytest.approx(conv["mean_f0"], abs=1e-6) == 0.0


def test_tama_synchrony(tama_files):
    wav_path, csv_path = tama_files
    features = ["mean_f0"]

    TA = TAMAProsodicAcommodation(
        audio_path=wav_path,
        transcript_csv=csv_path,
        requested_features=features,
        window_len=2.0,
        hop=2.0,
        verbose=False,
    )

    sync = TA.get_synchrony()

    # Sliding window on length=2 -> one correlation value for ([200,200] vs [400,400]) -> 0
    assert sync["mean_f0"].shape == (1,)
    assert pytest.approx(sync["mean_f0"][0], abs=1e-6) == 0.0