# Implementation Details for FastAPI-based ASR Application

## Implemented Features
- **Model Preparation**: Successfully loaded and exported the `stt_hi_conformer_ctc_medium` model to ONNX using NeMo’s export functionality.
- **FastAPI Endpoint**: Implemented `POST /transcribe` endpoint that accepts `.wav` files and returns transcribed text as JSON.
- **Input Validation**: Validates audio for 16kHz sample rate, mono channel, and 5–10 second duration.
- **Containerization**: Created a Dockerfile using Python 3.9-slim, exposing port 8000 and Streamlit (8501) via supervisord, with all dependencies installed.
- **Streamlit Frontend**: Interactive interface for uploading files, viewing transcripts, and downloading results.
- **Documentation**: Provided `README.md` with setup, usage, and testing instructions, and this `Description.md` for implementation details.

## Issues Encountered
- **ONNX Export Complexity**: Exporting the NeMo model to ONNX required careful handling of the model’s architecture, as Conformer-CTC models have complex layers. Simplified the process using NeMo’s built-in export method.
- **Inference Pipeline**: NeMo’s inference pipeline does not natively support async operations, leading to a synchronous implementation.
- **Streamlit Integration**: Required CORS and supervisord for dual servers.

## Limitations and Unimplemented Components
- **Async Inference**: Not implemented due to NeMo’s lack of async support. The synchronous pipeline may lead to higher latency under load.
  - **Reason**: NeMo’s inference API is designed for batch processing and doesn’t integrate well with Python’s `asyncio` for FastAPI.
  - **Solution**: Future work could involve wrapping the inference in a separate thread or process using `concurrent.futures`, though this adds complexity.
- **Model Quantization**: Omitted due to time constraints.
  - **Solution**:  Apply quantization post-submission.

## Overcoming Challenges
- **ONNX Export**: Use NeMo’s documentation and forums for troubleshooting export issues. Test with smaller audio clips to verify output.
- **Async Inference**: Explore process-based parallelism (e.g., `multiprocessing`) to simulate async behavior without blocking the FastAPI event loop.
- **Decoding Accuracy**: Implement beam search or use NeMo’s language model integration for better transcription quality.
- **Testing**: Develop test cases for audio validation and inference, and set up CI/CD to automate testing and deployment.

## Known Limitations and Assumptions
- **GPU Dependency**: Assumes access to a GPU for optimal inference speed; CPU fallback is slower.
- **Simplified Decoding**: Greedy decoding may miss nuances in Hindi transcription.
- **Single File Processing**: The endpoint processes one file at a time, limiting throughput for bulk requests.
- **Model Size**: The ONNX model is large (~500MB), increasing container size despite using a slim base image.
- **Language**: Assumes input audio is in Hindi, as the model is trained for Hindi speech.
