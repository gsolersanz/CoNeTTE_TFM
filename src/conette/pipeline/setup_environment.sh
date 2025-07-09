#!/bin/bash

# CoNeTTE T5 ONNX - Configuración Automática del Entorno
# ====================================================

set -e  # Exit on any error

echo "🚀 Configurando entorno CoNeTTE T5 ONNX Optimizado..."
echo "====================================================="

# Check if running on Jetson Nano
JETSON_NANO=false
if [ -f "/etc/nv_tegra_release" ]; then
    echo "🎯 Detectado Jetson Nano - Aplicando configuraciones específicas"
    JETSON_NANO=true
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activando entorno virtual..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Actualizando pip..."
pip install --upgrade pip

# Install base requirements
echo "📚 Instalando dependencias base..."
pip install -r requirements.txt

# Jetson Nano specific installations
if [ "$JETSON_NANO" = true ]; then
    echo "🎯 Instalando dependencias específicas para Jetson Nano..."
    
    # Install TensorRT-enabled ONNX Runtime
    pip install onnxruntime-gpu --extra-index-url https://developer.download.nvidia.com/compute/redist
    
    # Enable maximum performance mode
    echo "⚡ Habilitando modo performance máximo..."
    sudo nvpmodel -m 0
    sudo jetson_clocks
    
    # Set environment variables for optimal performance
    export CUDA_CACHE_DISABLE=0
    export CUDA_CACHE_MAXSIZE=2147483648
fi

# Create necessary directories
echo "📁 Creando estructura de directorios..."
mkdir -p 06_models/{onnx_models,onnx_models_optimized,t5_models,tokenizer_standalone}
mkdir -p logs
mkdir -p temp

# Download sample audio files for testing (if not exist)
echo "🎵 Descargando archivos de audio de prueba..."
mkdir -p sample_audio

# Create sample audio files info
cat > sample_audio/README.md << 'EOF'
# Sample Audio Files

Para probar el sistema, coloca archivos de audio (.wav) en esta carpeta.

Formatos soportados:
- WAV (recomendado)
- MP3
- FLAC

Ejemplos de nombres:
- voice.wav
- music.wav
- nature_sounds.wav
EOF

# Set up environment variables
echo "🔧 Configurando variables de entorno..."
cat > .env << 'EOF'
# CoNeTTE T5 ONNX Environment Variables
ONNX_MODELS_PATH=06_models/onnx_models
T5_MODELS_PATH=06_models/t5_models
TOKENIZER_PATH=06_models/tokenizer_standalone
SAMPLE_AUDIO_PATH=sample_audio
LOG_LEVEL=INFO
EOF

# Create quick test script
echo "🧪 Creando script de prueba rápida..."
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test script to verify installation."""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import torch
        print("✅ PyTorch disponible")
        
        import onnx
        print("✅ ONNX disponible")
        
        import onnxruntime as ort
        print("✅ ONNX Runtime disponible")
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        import librosa
        print("✅ Librosa disponible")
        
        import soundfile
        print("✅ SoundFile disponible")
        
        # Try to import CoNeTTE
        try:
            import conette
            print("✅ CoNeTTE disponible")
        except ImportError:
            print("⚠️ CoNeTTE no disponible - necesario solo para exportación inicial")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando dependencias: {e}")
        return False

def test_directories():
    """Test that all directories exist."""
    required_dirs = [
        "06_models",
        "sample_audio",
        "logs",
        "temp"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directorio {dir_path} existe")
        else:
            print(f"❌ Directorio {dir_path} no existe")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🧪 Probando instalación CoNeTTE T5 ONNX...")
    print("=" * 50)
    
    imports_ok = test_imports()
    dirs_ok = test_directories()
    
    if imports_ok and dirs_ok:
        print("\n✅ ¡Instalación completada exitosamente!")
        print("🚀 Puedes continuar con el uso del sistema")
    else:
        print("\n❌ Hay problemas con la instalación")
        print("Por favor revisa los errores arriba")
        sys.exit(1)
EOF

chmod +x quick_test.py

# Run quick test
echo "🧪 Ejecutando prueba de instalación..."
python quick_test.py

echo ""
echo "✅ ¡Configuración completada exitosamente!"
echo "========================================="
echo ""
echo "📋 Próximos pasos:"
echo "1. Activar entorno: source venv/bin/activate"
echo "2. Agregar audios de prueba en: sample_audio/"
echo "3. Ejecutar ejemplo rápido: python 08_examples/quick_start.py"
echo ""
echo "📚 Para más información consulta README.md"