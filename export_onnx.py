import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_hi_conformer_ctc_medium")

# Export with input names matching what ONNX expects
model.export(
    "stt_hi_conformer_ctc_medium.onnx"
)
