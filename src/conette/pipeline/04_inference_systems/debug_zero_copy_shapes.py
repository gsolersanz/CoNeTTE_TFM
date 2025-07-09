#!/usr/bin/env python3
"""
Debug extensivo para identificar el problema de dimensiones en zero-copy.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
import soundfile as sf
import librosa

def debug_audio_processing():
    """Debug detallado del procesamiento de audio."""
    
    print("🔍 DEBUG: Procesamiento de audio")
    print("=" * 50)
    
    # Cargar audio de prueba
    base_dir = Path(__file__).parent.parent.parent
    test_audio = str(base_dir / "data/sample.wav")
    
    if not os.path.exists(test_audio):
        print(f"❌ No se encontró: {test_audio}")
        return None
    
    # Procesar audio igual que en zero-copy
    audio, sr = sf.read(test_audio)
    print(f"📊 Audio original: shape={audio.shape}, sr={sr}")
    
    # Resamplear a 32000 Hz
    target_sr = 32000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Convertir a mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Normalizar
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    # Añadir batch dimension
    audio_batch = audio.reshape(1, -1).astype(np.float32)
    print(f"📊 Audio procesado: shape={audio_batch.shape}")
    
    return audio_batch

def debug_encoder_output(processed_audio):
    """Debug del encoder para obtener dimensiones exactas."""
    
    print("\n🔍 DEBUG: Salida del encoder")
    print("=" * 50)
    
    # Cargar encoder
    base_dir = Path(__file__).parent.parent.parent
    encoder_path = str(base_dir / "Definitivo/06_models/onnx_models/conette_encoder.onnx")
    
    if not os.path.exists(encoder_path):
        print(f"❌ No se encontró encoder: {encoder_path}")
        return None
    
    # Configurar sesión
    session_options = ort.SessionOptions()
    session_options.enable_cpu_mem_arena = True
    encoder_session = ort.InferenceSession(encoder_path, sess_options=session_options)
    
    # Preparar inputs
    audio_shape = np.array([[processed_audio.shape[1]]], dtype=np.int64)
    encoder_inputs = {
        'audio': processed_audio,
        'audio_shape': audio_shape
    }
    
    print(f"📊 Encoder inputs:")
    print(f"   - audio.shape: {processed_audio.shape}")
    print(f"   - audio_shape: {audio_shape}")
    
    # Ejecutar encoder
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    encoder_features = encoder_outputs[0]
    
    print(f"📊 Encoder outputs:")
    print(f"   - encoder_features.shape: {encoder_features.shape}")
    print(f"   - time_frames reales: {encoder_features.shape[2]}")
    
    # Calcular estimaciones
    estimated_frames_10240 = max(1, processed_audio.shape[1] // 10240)
    estimated_frames_400 = max(1, processed_audio.shape[1] // 400)  # Otra estimación común
    
    print(f"📊 Estimaciones:")
    print(f"   - frames // 10240: {estimated_frames_10240}")
    print(f"   - frames // 400: {estimated_frames_400}")
    print(f"   - frames reales: {encoder_features.shape[2]}")
    
    return encoder_features

def debug_zero_copy_binding(processed_audio, encoder_features):
    """Debug del binding zero-copy."""
    
    print("\n🔍 DEBUG: Zero-copy binding")
    print("=" * 50)
    
    # Cargar projection
    base_dir = Path(__file__).parent.parent.parent
    projection_path = str(base_dir / "Definitivo/06_models/onnx_models/conette_projection.onnx")
    
    if not os.path.exists(projection_path):
        print(f"❌ No se encontró projection: {projection_path}")
        return None
    
    # Configurar sesión
    session_options = ort.SessionOptions()
    session_options.enable_cpu_mem_arena = True
    projection_session = ort.InferenceSession(projection_path, sess_options=session_options)
    
    print(f"📊 Projection inputs esperados:")
    for i, input_info in enumerate(projection_session.get_inputs()):
        print(f"   - Input {i}: {input_info.name}, shape={input_info.shape}")
    
    print(f"📊 Projection outputs esperados:")
    for i, output_info in enumerate(projection_session.get_outputs()):
        print(f"   - Output {i}: {output_info.name}, shape={output_info.shape}")
    
    # Test estándar (sin zero-copy)
    try:
        projection_input_name = projection_session.get_inputs()[0].name
        projection_inputs = {projection_input_name: encoder_features}
        projection_outputs = projection_session.run(None, projection_inputs)
        projection_features = projection_outputs[0]
        
        print(f"✅ Projection estándar exitosa:")
        print(f"   - Input shape: {encoder_features.shape}")
        print(f"   - Output shape: {projection_features.shape}")
        
        return projection_features
        
    except Exception as e:
        print(f"❌ Error en projection estándar: {e}")
        return None

def debug_zero_copy_io_binding(encoder_features):
    """Debug específico del IOBinding."""
    
    print("\n🔍 DEBUG: IOBinding zero-copy")
    print("=" * 50)
    
    # Cargar projection
    base_dir = Path(__file__).parent.parent.parent
    projection_path = str(base_dir / "Definitivo/06_models/onnx_models/conette_projection.onnx")
    
    session_options = ort.SessionOptions()
    session_options.enable_cpu_mem_arena = True
    projection_session = ort.InferenceSession(projection_path, sess_options=session_options)
    
    # Crear IOBinding
    io_binding = projection_session.io_binding()
    
    try:
        # Bind input
        io_binding.bind_input(
            name=projection_session.get_inputs()[0].name,
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=encoder_features.shape,
            buffer_ptr=encoder_features.ctypes.data
        )
        
        # Prepare output buffer con tamaño correcto
        actual_time_frames = encoder_features.shape[2]
        projection_output_shape = [1, 256, actual_time_frames]
        projection_output = np.empty(projection_output_shape, dtype=np.float32)
        
        print(f"📊 IOBinding setup:")
        print(f"   - Input shape: {encoder_features.shape}")
        print(f"   - Output shape: {projection_output_shape}")
        
        # Bind output
        io_binding.bind_output(
            name=projection_session.get_outputs()[0].name,
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=projection_output_shape,
            buffer_ptr=projection_output.ctypes.data
        )
        
        # Run con IOBinding
        projection_session.run_with_iobinding(io_binding)
        
        print(f"✅ IOBinding exitoso:")
        print(f"   - Output shape: {projection_output.shape}")
        
        return projection_output
        
    except Exception as e:
        print(f"❌ Error en IOBinding: {e}")
        return None

def main():
    """Debug completo del sistema zero-copy."""
    
    print("🚀 DEBUG EXTENSIVO: T5 ONNX Zero-Copy")
    print("=" * 80)
    
    # 1. Debug audio processing
    processed_audio = debug_audio_processing()
    if processed_audio is None:
        return
    
    # 2. Debug encoder output
    encoder_features = debug_encoder_output(processed_audio)
    if encoder_features is None:
        return
    
    # 3. Debug projection estándar
    projection_features_std = debug_zero_copy_binding(processed_audio, encoder_features)
    if projection_features_std is None:
        return
    
    # 4. Debug IOBinding
    projection_features_iobinding = debug_zero_copy_io_binding(encoder_features)
    if projection_features_iobinding is None:
        return
    
    # 5. Comparar resultados
    print("\n🔍 COMPARACIÓN FINAL:")
    print("=" * 50)
    print(f"✅ Projection estándar: {projection_features_std.shape}")
    print(f"✅ Projection IOBinding: {projection_features_iobinding.shape}")
    
    # Verificar si son iguales
    if np.allclose(projection_features_std, projection_features_iobinding, atol=1e-6):
        print("✅ Resultados idénticos - IOBinding funciona correctamente")
    else:
        print("⚠️ Resultados diferentes - revisar IOBinding")
    
    print("\n🎯 SOLUCIÓN IDENTIFICADA:")
    print("=" * 50)
    print("El problema está en la estimación de time_frames.")
    print("Usar el encoder estándar primero para obtener dimensiones exactas.")

if __name__ == "__main__":
    main()