# FastAPI-based ASR Application with NVIDIA NeMo

This application serves an Automatic Speech Recognition (ASR) model using NVIDIA NeMo's `stt_hi_conformer_ctc_medium` model, optimized with ONNX, served via FastAPI and a Streamlit frontend.

## Prerequisites
- Docker installed
- A 5–10 second, 16kHz mono WAV audio file for testing
- Optional: GPU for faster inference (CPU works but is slower)



https://github.com/user-attachments/assets/fba9a3bb-8a4d-4f57-bf69-940548a294ad



## Build and Run
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Generate ONNX Model:**:
   ```bash
   python export_onnx.py
   ```

3. **Build the Docker Image**:
   ```bash
   docker build -t asr-app .
   ```

4. **Run the Container**:
   ```bash
   docker run -p 8000:8000 -p 8501:8501 asr-app
   ```

5. **Access the Application**:
   FastAPI: http://localhost:8000/docs
   Streamlit: http://localhost:8501

6. **Test the Endpoint**:
   Use `curl` or Postman to send a `.wav` file to the `/transcribe` endpoint:
   ```bash
   curl -X POST -F "file=@test_audio.wav" http://localhost:8000/transcribe
   ```

   **Sample Response**:
   ```json
   {
     "transcription": "नमस्ते यह एक टेस्ट है"
   }
   ```
7. **Use Streamlit**:
   Open http://localhost:8501.
   Upload one or multiple WAV files.
   View transcriptions and download results.

8. **Run Tests**:
   ```bash
   docker exec -it <container-id> pytest tests/test_main.py
   ```

## Design Considerations
- **Model Optimization**: The model is exported to ONNX for reduced inference latency.
- **Validation**: Ensures audio is 16kHz, mono, and 5–10 seconds.
- **Synchronous Inference**: Used due to NeMo’s lack of async support in the current version.
- **Lightweight Container**: Uses Python 3.9-slim to minimize image size.

## Notes
- Ensure the `stt_hi_conformer_ctc_medium.onnx` file is in the same directory as `main.py`.
- Test audio files must be 16kHz mono WAVs.
