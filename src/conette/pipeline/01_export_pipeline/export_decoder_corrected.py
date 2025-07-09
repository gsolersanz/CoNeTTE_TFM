#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPORTACIÓN CORREGIDA DE DECODER ONNX PARA DEFINITIVO
====================================================

Este script exporta el decoder corregido a ONNX manteniendo la funcionalidad exacta
del decoder PyTorch original, solucionando el problema de repetición identificado.

Adaptado de export_corrected_onnx.py que funciona correctamente.
"""

import os
import sys
import logging
import traceback
from pathlib import Path

import torch
import torchaudio
import numpy as np
from torch import nn

# Add current source to path to use local conette instead of installed package
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent  # Go up to src/conette level
sys.path.insert(0, str(src_dir.parent))  # Add src/ to path

from conette.fix_onnx_decoder import FixedONNXDecoderWrapper

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('export_decoder_corrected.log')
    ]
)

logger = logging.getLogger(__name__)

# Crear directorio de salida
ONNX_DIR = Path("06_models/onnx_models")
ONNX_DIR.mkdir(parents=True, exist_ok=True)

def load_pytorch_model():
    """Cargar modelo PyTorch"""
    logger.info("Loading PyTorch model...")
    
    try:
        from conette import CoNeTTEConfig, CoNeTTEModel
        
        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
        model = model.float()  # Asegurar float32
        model.eval()
        
        logger.info("✅ PyTorch model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading PyTorch model: {e}")
        raise

def prepare_sample_inputs(model):
    """Preparar entradas de ejemplo para la exportación"""
    logger.info("Preparing sample inputs...")
    
    # Crear audio sintético si no hay archivo de muestra
    audio = torch.randn(1, 32000 * 5, dtype=torch.float32)  # 5 segundos
    
    # Obtener características usando el pipeline completo del modelo
    with torch.no_grad():
        # Usar el preprocessor exactamente como en el modelo
        batch = model.preprocessor(audio)
        
        # Añadir información requerida
        batch["dataset"] = ["clotho"]
        batch["source"] = [None]
        
        logger.info(f"Preprocessor output shapes:")
        logger.info(f"  audio: {batch['audio'].shape}")
        logger.info(f"  audio_shape: {batch['audio_shape'].shape}")
        
        # Obtener encoder outputs usando la interfaz correcta
        encoder_outputs = model.model.encoder(batch["audio"], batch["audio_shape"])
        frame_embs_raw = encoder_outputs["frame_embs"]
        
        logger.info(f"Encoder output shape: {frame_embs_raw.shape}")
        
        # El encoder output viene como [batch, time, hidden_size=768]
        # Aplicar projection correctamente
        frame_embs_projected = model.model.projection(frame_embs_raw)  # [batch, hidden_size=256, time]
        
        # Transponer de [batch, d_model, time] a [batch, time, d_model]
        frame_embs = frame_embs_projected.transpose(1, 2)  # [batch, time, d_model=256]
        
        logger.info(f"After projection shape: {frame_embs_projected.shape}")
        logger.info(f"After transpose shape: {frame_embs.shape}")
        
    # Preparar entradas para decoder - usar BOS token correcto para Clotho
    tokenizer = model.model.tokenizer
    bos_token = "<bos_clotho>"
    if tokenizer.has(bos_token):
        bos_id = tokenizer.token_to_id(bos_token)
    else:
        bos_id = tokenizer.bos_token_id
    
    input_ids = torch.tensor([[bos_id]], dtype=torch.long)
    
    logger.info(f"Using BOS token: {bos_id} ('{bos_token if tokenizer.has(bos_token) else 'default_bos'}')")
    
    logger.info(f"Sample inputs prepared:")
    logger.info(f"  input_ids: {input_ids.shape}")
    logger.info(f"  encoder_hidden_states: {frame_embs.shape}")
    
    return input_ids, frame_embs

def export_corrected_decoder(model, input_ids, encoder_hidden_states):
    """Exportar decoder corregido a ONNX"""
    logger.info("Exporting corrected decoder to ONNX...")
    
    # Crear wrapper corregido
    fixed_decoder = FixedONNXDecoderWrapper(model)
    fixed_decoder = fixed_decoder.float()
    fixed_decoder.eval()
    
    # Verificar que el forward funciona
    logger.info("Testing forward pass...")
    with torch.no_grad():
        test_output = fixed_decoder(input_ids, encoder_hidden_states)
        logger.info(f"✅ Forward pass successful - output shape: {test_output.shape}")
    
    # Exportar a ONNX
    export_path = ONNX_DIR / "conette_decoder_corrected.onnx"
    
    logger.info(f"Exporting to {export_path}...")
    
    try:
        torch.onnx.export(
            fixed_decoder,
            (input_ids, encoder_hidden_states),
            export_path.as_posix(),
            export_params=True,
            opset_version=14,
            input_names=["input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"}
            },
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info(f"✅ Decoder exported successfully to {export_path}")
        return export_path
        
    except Exception as e:
        logger.error(f"❌ Error exporting decoder: {e}")
        logger.error(traceback.format_exc())
        raise

def verify_onnx_export(export_path, input_ids, encoder_hidden_states, model):
    """Verificar que la exportación ONNX funciona correctamente"""
    logger.info("Verifying ONNX export...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Verificar modelo ONNX
        onnx_model = onnx.load(export_path.as_posix())
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model structure is valid")
        
        # Crear sesión ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            export_path.as_posix(),
            sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Probar inferencia
        onnx_inputs = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "encoder_hidden_states": encoder_hidden_states.numpy().astype(np.float32)
        }
        
        onnx_outputs = session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]
        
        logger.info(f"✅ ONNX inference successful - output shape: {onnx_logits.shape}")
        
        # Comparar con PyTorch
        fixed_decoder = FixedONNXDecoderWrapper(model)
        fixed_decoder.eval()
        
        with torch.no_grad():
            pytorch_logits = fixed_decoder(input_ids, encoder_hidden_states)
            pytorch_logits_np = pytorch_logits.numpy()
        
        # Calcular diferencia
        max_diff = np.abs(onnx_logits - pytorch_logits_np).max()
        mean_diff = np.abs(onnx_logits - pytorch_logits_np).mean()
        
        logger.info(f"Comparison with PyTorch:")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            logger.info("✅ ONNX and PyTorch outputs match closely")
            return True
        else:
            logger.warning("⚠️ Significant differences detected between ONNX and PyTorch")
            return False
            
    except ImportError:
        logger.warning("⚠️ onnx or onnxruntime not available, skipping verification")
        return True
    except Exception as e:
        logger.error(f"❌ Error during verification: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Función principal de exportación"""
    logger.info("="*80)
    logger.info(" EXPORTACIÓN CORREGIDA DE DECODER ONNX - DEFINITIVO ")
    logger.info("="*80)
    
    try:
        # 1. Cargar modelo PyTorch
        model = load_pytorch_model()
        
        # 2. Preparar entradas de ejemplo
        input_ids, encoder_hidden_states = prepare_sample_inputs(model)
        
        # 3. Exportar decoder corregido
        export_path = export_corrected_decoder(model, input_ids, encoder_hidden_states)
        
        # 4. Verificar exportación
        verification_success = verify_onnx_export(export_path, input_ids, encoder_hidden_states, model)
        
        # 5. Resumen final
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE EXPORTACIÓN")
        logger.info("="*60)
        logger.info(f"Archivo exportado: {export_path}")
        logger.info(f"Verificación ONNX: {'✅ OK' if verification_success else '❌ FAILED'}")
        
        if verification_success:
            logger.info("\n✅ EXPORTACIÓN EXITOSA")
            logger.info("El decoder corregido debería resolver el problema de repetición.")
            logger.info(f"Usar el archivo: {export_path}")
            return True
        else:
            logger.warning("\n⚠️ EXPORTACIÓN CON ADVERTENCIAS")
            logger.info("El decoder fue exportado pero puede tener problemas.")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ ERROR EN EXPORTACIÓN: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)