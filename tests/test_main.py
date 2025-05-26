import pytest
import numpy as np
from fastapi.testclient import TestClient
from main import app, validate_audio
import torchaudio
import os
import requests
import torch

client = TestClient(app)

def test_validate_audio_valid():
    """Test audio validation with a valid file."""
    waveform = np.random.randn(1, 160000)  # 10 seconds at 16kHz
    temp_path = "test_valid.wav"
    torchaudio.save(temp_path, torch.tensor(waveform), 16000)
    try:
        audio = validate_audio(temp_path)
        assert audio.shape[0] == 1
        assert audio.shape[1] == 160000
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_validate_audio_invalid_duration():
    """Test audio validation with invalid duration."""
    waveform = np.random.randn(1, 32000)  # 2 seconds
    temp_path = "test_invalid.wav"
    torchaudio.save(temp_path, torch.tensor(waveform), 16000)
    try:
        with pytest.raises(Exception, match="Audio duration must be between 5 and 10 seconds"):
            validate_audio(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_transcribe_endpoint():
    """Test the /transcribe endpoint with a valid file."""
    waveform = np.random.randn(1, 160000)
    temp_path = "test_audio.wav"
    torchaudio.save(temp_path, torch.tensor(waveform), 16000)
    with open(temp_path, "rb") as f:
        response = client.post("/transcribe", files=[("files", ("test_audio.wav", f, "audio/wav"))])
    os.remove(temp_path)
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert "transcription" in response.json()[0]
    assert "confidence" in response.json()[0]

def test_transcribe_batch_endpoint():
    """Test the /transcribe endpoint with multiple files."""
    waveform = np.random.randn(1, 160000)
    temp_paths = ["test_audio1.wav", "test_audio2.wav"]
    for path in temp_paths:
        torchaudio.save(path, torch.tensor(waveform), 16000)
    files = [
        ("files", ("test_audio1.wav", open(temp_paths[0], "rb"), "audio/wav")),
        ("files", ("test_audio2.wav", open(temp_paths[1], "rb"), "audio/wav")),
    ]
    response = client.post("/transcribe", files=files)
    for path in temp_paths:
        os.remove(path)
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert all("transcription" in result for result in response.json())

def test_streamlit_api():
    """Test Streamlit frontend API call to /transcribe."""
    waveform = np.random.randn(1, 160000)
    temp_path = "test_audio.wav"
    torchaudio.save(temp_path, torch.tensor(waveform), 16000)
    try:
        with open(temp_path, "rb") as f:
            files = [("files", ("test_audio.wav", f, "audio/wav"))]
            response = requests.post("http://localhost:8000/transcribe", files=files)
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert "transcription" in response.json()[0]
    finally:
        os.remove(temp_path)