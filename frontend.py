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
                st.write(f"Debug: type of results: {type(results)}")
                st.write(f"Debug: results content: {results}")
                if isinstance(results, list):
                    for result in results:
                        st.success(f"Transcription: {results}")
                        st.download_button(
                            label=f"Download Transcription",
                            data=results,
                            file_name=f"transcription.txt",
                            mime="text/plain",
                        )
                elif isinstance(results, dict):
                    # Handle single result dict
                    st.success(f"Transcription: {results.get('transcriptions', '')}")
                    # st.download_button(
                    #     label=f"Download Transcription",
                    #     data=results.get('transcriptions', ''),
                    #     file_name=f"transcription.txt",
                    #     mime="text/plain",
                    # )
                else:
                    st.error(f"Unexpected response format: {results}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")

st.info("Note: Audio files must be 16kHz, mono, and 5–10 seconds long. Ensure the FastAPI server is running on port 8000.")
