import os
import pytest
from audio_features.audio_features import AudioFeatures


current_directory = os.path.dirname(__file__)
TEST_WAV = os.path.join(current_directory, "data", "example.wav")

@pytest.fixture
def audio_feature_instance():
    return AudioFeatures(TEST_WAV, "en")

def test_get_average_pitch(audio_feature_instance):
    result = audio_feature_instance.get_average_pitch()
    assert isinstance(result, dict)
    assert "average_pitch" in result
    assert result["average_pitch"] > 0

def test_get_f0_statistics(audio_feature_instance):
    result = audio_feature_instance.get_f0_statistics()
    assert isinstance(result, dict)
    assert "mean_f0" in result
    assert result["mean_f0"] > 0

def test_get_cumulative_speech_index(audio_feature_instance):
    result = audio_feature_instance.get_cumulative_speech_index()
    assert isinstance(result, dict)
    assert "csi" in result
    assert result["csi"] >= 0

def test_get_vocal_intensity_statistics(audio_feature_instance):
    result = audio_feature_instance.get_vocal_intensity_statistics()
    assert isinstance(result, dict)
    assert "mean_intensity" in result
    assert result["mean_intensity"] > 0

def test_get_all_features(audio_feature_instance):
    result = audio_feature_instance.get_all_features()
    assert isinstance(result, dict)
    assert "average_pitch" in result

import numpy as np
import pytest
import soundfile as sf
import os

from audio_features import AudioFeatures

def write_sine_wav(filename, freq, duration=2.0, sr=16000, amplitude=0.1):
    """
    Helper: write a mono WAV of a pure sine wave at `freq` Hz, duration `duration` sec.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t).astype("float32")
    sf.write(filename, wave, sr)

@pytest.fixture
def sine_wav(tmp_path):
    wav_path = tmp_path / "sine.wav"
    write_sine_wav(str(wav_path), freq=200, duration=2.0, sr=16000, amplitude=0.1)
    return str(wav_path)

def test_extract_mean_f0_array():
    """
    Passing a raw NumPy array (pure 200 Hz tone) should yield mean_f0 ≈ 200 Hz.
    """
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    array = 0.1 * np.sin(2 * np.pi * 200 * t).astype("float32")

    af = AudioFeatures(array=array, sr=sr)
    feats = af.extract(["mean_f0", "sd_f0"], verbose=False)

    assert "mean_f0" in feats
    assert "sd_f0" in feats
    # mean_f0 should be approx 200 Hz within 5%
    assert pytest.approx(feats["mean_f0"], rel=0.05) == 200.0
    # A pure tone has very low pitch variance
    assert feats["sd_f0"] < 1.0

def test_extract_mean_f0_file(sine_wav):
    """
    Loading from a 200 Hz sine‐wave WAV file should give mean_f0 ≈ 200 Hz,
    plus nonzero mean_intensity and near‐zero CSI.
    """
    af = AudioFeatures(path=sine_wav)
    feats = af.extract(
        ["mean_f0", "sd_f0", "mean_intensity", "csi"],
        verbose=False
    )

    # Basic F0 checks
    assert "mean_f0" in feats and "sd_f0" in feats
    assert pytest.approx(feats["mean_f0"], rel=0.05) == 200.0

    # Intensity must be a finite number (usually > 0 dB)
    assert np.isfinite(feats["mean_intensity"])

    # CSI for a perfect sine tone should be near zero (no pitch variation)
    assert abs(feats["csi"]) < 1e-6

def test_unrecognized_feature_returns_zero(sine_wav):
    """
    If we request a feature name not in the supported list,
    AudioFeatures.extract(...) should return it with value 0.0.
    """
    af = AudioFeatures(path=sine_wav)
    feats = af.extract(["nonexistent_feature"], verbose=False)

    assert "nonexistent_feature" in feats
    assert feats["nonexistent_feature"] == 0.0
