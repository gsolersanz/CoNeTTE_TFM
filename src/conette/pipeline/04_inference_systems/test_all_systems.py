#!/usr/bin/env python3
"""
Test all optimized inference systems to verify they work correctly.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce verbosity
logger = logging.getLogger(__name__)

def test_system(system_name, module_name):
    """Test a specific inference system."""
    print(f"\n🧪 TESTING: {system_name}")
    print("=" * 60)
    
    try:
        # Import the system
        sys.path.insert(0, str(Path(__file__).parent))
        module = __import__(module_name)
        
        # Get the main class
        if hasattr(module, 'T5ONNXStandaloneInference'):
            SystemClass = module.T5ONNXStandaloneInference
        elif hasattr(module, 'T5ONNXCorrected'):
            SystemClass = module.T5ONNXCorrected
        elif hasattr(module, 'T5ONNXZeroCopyInference'):
            SystemClass = module.T5ONNXZeroCopyInference
        elif hasattr(module, 'T5ONNXFP16Inference'):
            SystemClass = module.T5ONNXFP16Inference
        elif hasattr(module, 'T5ONNXFullyOptimized'):
            SystemClass = module.T5ONNXFullyOptimized
        else:
            print("❌ No se encontró clase principal")
            return False
        
        # Initialize system with paths relative to conette directory
        start_time = time.time()
        
        # Check which parameters the class accepts
        import inspect
        init_signature = inspect.signature(SystemClass.__init__)
        params = list(init_signature.parameters.keys())
        
        # Base configuration - paths are relative to conette directory
        config = {
            "decoder_path": "Definitivo/06_models/t5_models/dec_no_cache/model.onnx",
            "encoder_path": "Definitivo/06_models/onnx_models/conette_encoder.onnx",
            "projection_path": "Definitivo/06_models/onnx_models/conette_projection.onnx",
            "tokenizer_dir": "Definitivo/06_models/conette_tokenizer_standalone"
        }
        
        # Add optional parameters only if the class accepts them
        optional_params = {
            "enable_zero_copy": True,
            "enable_fp16": True,
            "enable_tensorrt": True,
            "jetson_nano_mode": True
        }
        
        for param, value in optional_params.items():
            if param in params:
                config[param] = value
        
        # Create instance with filtered parameters
        system = SystemClass(**config)
        init_time = time.time() - start_time
        print(f"✅ Sistema inicializado en {init_time:.2f}s")
        
        # Test with audio files - use paths relative to conette directory
        base_dir = Path(__file__).parent.parent.parent  # Go back to conette/
        test_audio_candidates = [
            "data/voice.wav",
            "data/sample.wav",
            "data/Bicycle_Bell.wav",
            "Definitivo/data/voice.wav",
            "Definitivo/data/sample.wav"
        ]
        
        test_audio = None
        for candidate in test_audio_candidates:
            full_path = str(base_dir / candidate)
            if os.path.exists(full_path):
                test_audio = full_path
                print(f"📁 Usando archivo de audio: {candidate}")
                break
        
        if test_audio and os.path.exists(test_audio):
            result = system.predict(test_audio)
            
            if result['success']:
                print(f"✅ Caption: '{result['caption']}'")
                print(f"✅ Tiempo: {result['total_time']:.3f}s")
                return True
            else:
                print(f"❌ Error: {result['error']}")
                return False
        else:
            print("❌ Archivo de audio de prueba no encontrado")
            print("   Archivos buscados:")
            for candidate in test_audio_candidates:
                full_path = str(base_dir / candidate)
                print(f"   - {candidate}: {'✓' if os.path.exists(full_path) else '✗'}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all systems."""
    print("🚀 TESTING ALL OPTIMIZED INFERENCE SYSTEMS")
    print("=" * 80)
    
    # Get base directory for displaying paths
    base_dir = Path(__file__).parent.parent.parent  # Go back to conette/
    
    print("\n📁 Configuración de paths (relativos a conette/):")
    print(f"   - Decoder: Definitivo/06_models/t5_models/dec_no_cache/model.onnx")
    print(f"   - Encoder: Definitivo/06_models/onnx_models/conette_encoder.onnx")
    print(f"   - Projection: Definitivo/06_models/onnx_models/conette_projection.onnx")
    print(f"   - Tokenizer: Definitivo/06_models/conette_tokenizer_standalone")
    print(f"   - Base directory: {base_dir}")
    print(f"   - Configuración: zero_copy=True, fp16=True, tensorrt=True, jetson_nano=True\n")
    
    systems = [
        ("T5 ONNX Standalone", "t5_onnx_standalone_inference"),
        ("T5 ONNX Corrected", "t5_onnx_inference"),
        ("T5 ONNX Zero-Copy", "t5_onnx_zero_copy_inference"),
        ("T5 ONNX FP16", "t5_onnx_fp16_inference"),
        ("T5 ONNX Fully Optimized", "t5_onnx_fully_optimized_fixed")
    ]
    
    results = {}
    
    for system_name, module_name in systems:
        success = test_system(system_name, module_name)
        results[system_name] = success
    
    # Summary
    print(f"\n📋 RESUMEN FINAL:")
    print("=" * 80)
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"✅ Sistemas funcionando: {len(successful)}/{len(systems)}")
    for name in successful:
        print(f"   ✅ {name}")
    
    if failed:
        print(f"\n❌ Sistemas con problemas: {len(failed)}")
        for name in failed:
            print(f"   ❌ {name}")
    
    print(f"\n🎯 Estado: {'TODOS FUNCIONANDO' if len(successful) == len(systems) else 'ALGUNOS NECESITAN CORRECCIÓN'}")

if __name__ == "__main__":
    main()