#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
from pathlib import Path
import soundfile as sf
import librosa

def calculate_exact_time_frames(audio_length):
    """Calculate exact time frames that the encoder will output."""
    # Based on debugging multiple audio files, let's derive the formula
    
    # Test with different audio lengths to find the pattern
    test_lengths = [
        (828860, 81),  # voice.wav
        (920000, 90),  # sample.wav  
        # Need to add more data points
    ]
    
    # For now, use a more precise calculation
    # The encoder seems to use some downsampling factor
    # Let's run a few tests to find the exact formula
    
    # Rough estimate based on samples:
    # 828860 → 81 frames  ≈ 10230 samples per frame
    # 920000 → 90 frames  ≈ 10222 samples per frame
    
    # More conservative estimate
    time_frames = max(1, audio_length // 10240)
    
    return time_frames

def test_encoder_with_different_sizes():
    """Test encoder with different audio sizes to understand the pattern."""
    
    base_dir = Path(__file__).parent.parent.parent
    encoder_path = str(base_dir / "onnx_models_full/conette_encoder.onnx")
    session = ort.InferenceSession(encoder_path)
    
    # Test with different audio files
    test_files = [
        "/workspace/conette/data/voice.wav",
        "/workspace/conette/data/sample.wav", 
        "/workspace/conette/data/Bicycle_Bell.wav"
    ]
    
    results = []
    
    for audio_path in test_files:
        try:
            # Load and preprocess
            audio, sr = sf.read(audio_path)
            target_sr = 32000
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio / (np.abs(audio).max() + 1e-8)
            audio_batch = audio.reshape(1, -1).astype(np.float32)
            
            # Run encoder
            audio_shape = np.array([[audio_batch.shape[1]]], dtype=np.int64)
            encoder_inputs = {'audio': audio_batch, 'audio_shape': audio_shape}
            encoder_outputs = session.run(None, encoder_inputs)
            output_shape = encoder_outputs[0].shape
            
            # Calculate ratio
            ratio = audio_batch.shape[1] / output_shape[2]
            
            results.append({
                'file': audio_path.split('/')[-1],
                'audio_length': audio_batch.shape[1],
                'time_frames': output_shape[2],
                'ratio': ratio,
                'full_shape': output_shape
            })
            
            print(f"File: {audio_path.split('/')[-1]}")
            print(f"  Audio length: {audio_batch.shape[1]}")
            print(f"  Output shape: {output_shape}")
            print(f"  Time frames: {output_shape[2]}")
            print(f"  Ratio: {ratio:.2f}")
            print()
            
        except Exception as e:
            print(f"Error with {audio_path}: {e}")
    
    # Analyze the pattern
    if len(results) >= 2:
        print("PATTERN ANALYSIS:")
        ratios = [r['ratio'] for r in results]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Average ratio: {avg_ratio:.2f}")
        print(f"Min ratio: {min(ratios):.2f}")
        print(f"Max ratio: {max(ratios):.2f}")
        
        # Suggest formula
        print(f"\nSUGGESTED FORMULA:")
        print(f"time_frames = audio_length // {int(avg_ratio)}")
    
    return results

if __name__ == "__main__":
    test_encoder_with_different_sizes()