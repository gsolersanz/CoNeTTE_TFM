#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
from pathlib import Path
import soundfile as sf
import librosa

# Test the encoder output shape
base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
encoder_path = str(base_dir / "onnx_models_full/conette_encoder.onnx")

print(f"Testing encoder: {encoder_path}")

# Load encoder
session = ort.InferenceSession(encoder_path)

# Test with real audio
audio_path = "/workspace/conette/data/voice.wav"
audio, sr = sf.read(audio_path)

# Preprocess like working system
target_sr = 32000
if sr != target_sr:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Normalize
audio = audio / (np.abs(audio).max() + 1e-8)

# Add batch dimension
audio_batch = audio.reshape(1, -1).astype(np.float32)
print(f"Input audio shape: {audio_batch.shape}")

# Audio shape for encoder
audio_shape = np.array([[audio_batch.shape[1]]], dtype=np.int64)
print(f"Audio shape input: {audio_shape.shape} = {audio_shape}")

# Run encoder
encoder_inputs = {
    'audio': audio_batch,
    'audio_shape': audio_shape
}

try:
    encoder_outputs = session.run(None, encoder_inputs)
    output = encoder_outputs[0]
    print(f"Encoder output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Calculate time dimension
    time_frames = output.shape[-1] if len(output.shape) == 3 else output.shape[1]
    print(f"Time frames: {time_frames}")
    
    # For zero-copy, we need to know the exact output shape
    print(f"Zero-copy output shape should be: {list(output.shape)}")
    
except Exception as e:
    print(f"Error: {e}")
    
    # Check inputs and outputs from model
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    
    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")