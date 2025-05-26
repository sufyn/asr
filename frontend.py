import streamlit as st
import requests
import os
from typing import List, Dict
import io

st.set_page_config(page_title="Hindi ASR Transcription App", page_icon="🎙️")

st.title("🎙️ Hindi ASR Transcription App")
st.write("Upload 5–10 second, 16kHz mono WAV files to transcribe Hindi audio using NVIDIA NeMo.")

# Single or batch file upload
st.header("Transcribe Audio Files")
uploaded_files = st.file_uploader("Choose WAV file(s)", type=["wav"], accept_multiple_files=True)
if uploaded_files:
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            files = [(f"files", (file.name, file, "audio/wav")) for file in uploaded_files]
            try:
                response = requests.post("http://localhost:8000/transcribe", files=files)
                response.raise_for_status()
                results = response.json()
                for result in results:
                    st.write(f"**File**: {result['filename']}")
                    st.success(f"Transcription: {result['transcription']}")
                    # st.write(f"Confidence: {result['confidence']:.2%}")
                    st.download_button(
                        label=f"Download {result['filename']} Transcription",
                        data=result['transcription'],
                        file_name=f"{result['filename']}_transcription.txt",
                        mime="text/plain",
                    )
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")

st.info("Note: Audio files must be 16kHz, mono, and 5–10 seconds long. Ensure the FastAPI server is running on port 8000.")