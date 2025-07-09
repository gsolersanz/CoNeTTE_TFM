#!/usr/bin/env python3

import numpy as np
import torch
import sys
from pathlib import Path

# Add fallback path for conette import  
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from conette.huggingface.model import CoNeTTEModel
from conette.huggingface.config import CoNeTTEConfig

print("DEBUG: Checking audio preprocessing format")
print("=" * 50)

# Load preprocessor
config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
preprocessor = model.preprocessor

# Test with dummy audio
print("Testing with dummy audio...")
dummy_audio = np.random.randn(32000).astype(np.float32)  # 1 second at 32kHz
audio_tensor = torch.from_numpy(dummy_audio).float().unsqueeze(0)  # [1, time_steps]
print(f"Input audio tensor shape: {audio_tensor.shape}")

with torch.no_grad():
    batch = preprocessor(audio_tensor)
    
    print(f"Preprocessor output keys: {list(batch.keys())}")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    # Check specifically the audio field
    audio_output = batch['audio']
    print(f"\nAudio output analysis:")
    print(f"  Shape: {audio_output.shape}")
    print(f"  Rank: {len(audio_output.shape)}")
    print(f"  Dtype: {audio_output.dtype}")
    
    # Convert to numpy
    audio_numpy = audio_output.numpy()
    print(f"  As numpy: {audio_numpy.shape} ({audio_numpy.dtype})")

print("\nNow checking how working system processes this...")

# Check what working system expects
print("\nComparing with working t5_onnx_inference_pytorch_tokenizer.py:")
import soundfile as sf
import librosa

# Try to load real audio file if available
audio_files = [
    "/workspace/conette/data/voice.wav",
    "/workspace/conette/data/sample.wav"
]

for audio_file in audio_files:
    try:
        audio, sr = sf.read(audio_file)
        target_sr = 32000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
            
        print(f"\nReal audio file: {audio_file}")
        print(f"  Raw audio shape: {audio.shape}")
        
        # Test with preprocessor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        print(f"  As tensor: {audio_tensor.shape}")
        
        with torch.no_grad():
            batch = preprocessor(audio_tensor)
            processed = batch['audio']
            print(f"  Preprocessed: {processed.shape}")
            
        break
    except Exception as e:
        print(f"Failed to load {audio_file}: {e}")
        continue