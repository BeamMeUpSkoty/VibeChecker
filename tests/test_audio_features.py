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
