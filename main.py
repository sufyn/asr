from fastapi import FastAPI, UploadFile, File
from typing import List
import shutil
import os
import uuid
import torchaudio
import torch
import nemo.collections.asr as nemo_asr
from concurrent.futures import ThreadPoolExecutor
import uvicorn

app = FastAPI()

# Load the model once globally
hf_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_hi_conformer_ctc_medium")
device = 'cpu'  # Change to 'cuda' if GPU available

# Your original transcribe_single_audio function
def transcribe_single_audio(audio_file, model=hf_model, device='cpu'):
    waveform, sample_rate = torchaudio.load(audio_file)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    with torch.no_grad():
        waveform = waveform.to(device)
        model = model.to(device)
        logits, logits_len, _ = model.forward(input_signal=waveform, input_signal_length=torch.tensor([waveform.size(1)]).to(device))
        current_hypotheses = model.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_len, return_hypotheses=True,
        )
        text = current_hypotheses[0].text
    return text

# Wrapper for multiple files (if needed)
def transcribe_audio(audio_files, model=hf_model, device='cpu', max_workers=None):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(transcribe_single_audio, audio_file, model, device) for audio_file in audio_files]
        results = [future.result() for future in futures]
    return results

# FastAPI endpoint
@app.post("/transcribe")
async def transcribe_audio_files(files: List[UploadFile] = File(...)):
    # Save uploaded files temporarily
    saved_paths = []
    for file in files:
        filename = f"temp_{uuid.uuid4().hex}.wav"
        with open(filename, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(filename)

    # Transcribe using your original logic
    transcriptions = transcribe_audio(saved_paths, device=device)

    # Delete temp files
    for path in saved_paths:
        os.remove(path)

    return {"transcriptions": transcriptions}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
