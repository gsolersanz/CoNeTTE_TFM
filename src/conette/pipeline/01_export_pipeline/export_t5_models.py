#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPORTADOR T5 MODELS PARA CoNeTTE - VERSIÓN FUNCIONAL
====================================================

Este script exporta el decoder CoNeTTE adaptado a formato T5 a ONNX usando
la implementación probada y funcional de conette_t5_approach.py

MODELO EXPORTADO:
- model.onnx: Decoder T5-compatible para uso con encoder CoNeTTE

DIRECTORIO DE SALIDA: 06_models/t5_models/dec_no_cache/
RUTA FUNCIONAL GENERADA: conette_t5/dec_no_cache/model.onnx (enlace simbólico)

ARQUITECTURA:
Audio → ONNX Encoder → ONNX Projection → T5 Decoder ONNX → Beam Search
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any

import torch
import torchaudio
import numpy as np
from torch import nn, Tensor
import onnx
import onnxruntime as ort

# Add current source to path to use local conette instead of installed package
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent  # Go up to src/conette level
sys.path.insert(0, str(src_dir.parent))  # Add src/ to path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('export_t5_models.log')
    ]
)

logger = logging.getLogger(__name__)

# Constantes
T5_MODELS_DIR = Path("06_models/t5_models/dec_no_cache")
T5_FUNCTIONAL_DIR = Path("conette_t5/dec_no_cache")  # Directorio funcional para enlaces

# Crear directorios
T5_MODELS_DIR.mkdir(parents=True, exist_ok=True)
T5_FUNCTIONAL_DIR.mkdir(parents=True, exist_ok=True)

class ExportCoNeTTE(nn.Module):
    """
    Wrapper de CoNeTTE equivalente al ExportT5, adaptado para exportación T5.
    Implementación funcional probada de conette_t5_approach.py
    """
    
    def __init__(self, model):
        super().__init__()
        
        # Extraer componentes del modelo CoNeTTE
        self.decoder = model.model.decoder
        
        # En CoNeTTE, el classifier ya está dentro del decoder
        # Pero necesitamos acceso directo para el wrapper
        self.classifier = self.decoder.classifier
        
        # Parámetros importantes
        self.bos_id = self.decoder.bos_id  
        self.eos_id = self.decoder.eos_id
        self.pad_id = self.decoder.pad_id
        self.vocab_size = self.decoder.vocab_size
        
        # Obtener d_model del embedding layer
        if hasattr(self.decoder, 'd_model'):
            self.d_model = self.decoder.d_model
        else:
            self.d_model = self.decoder.emb_layer.embedding_dim
            logger.info(f"d_model inferido del embedding: {self.d_model}")
        
        logger.info(f"ExportCoNeTTE inicializado:")
        logger.info(f"  - d_model: {self.d_model}")
        logger.info(f"  - vocab_size: {self.vocab_size}")
        logger.info(f"  - BOS: {self.bos_id}, EOS: {self.eos_id}")
    
    def forward(self, 
                input_ids: Tensor,
                encoder_hidden_states: Tensor, 
                past_key_values: Optional[Tuple] = None):
        """
        Forward que replica la interfaz de T5 pero para CoNeTTE.
        Implementación funcional probada.
        """
        
        logger.debug(f"ExportCoNeTTE forward:")
        logger.debug(f"  input_ids: {input_ids.shape}")
        logger.debug(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
        logger.debug(f"  past_key_values: {'None' if past_key_values is None else 'presente'}")
        
        # Preparar inputs para el decoder CoNeTTE
        batch_size, seq_len = input_ids.shape
        
        # encoder_hidden_states viene en formato [batch, seq_len, hidden_size]
        # CoNeTTE decoder espera [seq_len, batch, hidden_size]
        frame_embs = encoder_hidden_states.transpose(0, 1)  # [seq_len, batch, hidden_size]
        
        # input_ids está en [batch, seq_len]
        # CoNeTTE decoder espera [seq_len, batch]  
        caps_in = input_ids.transpose(0, 1)  # [seq_len, batch]
        
        # Crear máscaras (CoNeTTE las necesita)
        frame_embs_pad_mask = None  # Por simplicidad, no usar mask en encoder
        caps_in_pad_mask = None     # Por simplicidad, no usar mask en decoder
        
        # Crear causal mask para decoder (obligatorio)
        device = input_ids.device
        caps_in_sq_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), 
            diagonal=1
        )
        
        # Usar el decoder original de CoNeTTE
        if past_key_values is None:
            # Primer paso: usar decoder completo
            logger.debug("  -> Primer paso (sin cache)")
            
            # Llamar al decoder original de CoNeTTE
            logits = self.decoder.forward(
                frame_embs=frame_embs,
                frame_embs_pad_mask=frame_embs_pad_mask,
                caps_in=caps_in,
                caps_in_pad_mask=caps_in_pad_mask,
                caps_in_sq_mask=caps_in_sq_mask,
            )
            
            # logits viene en formato [seq_len, batch, vocab_size]
            # Necesitamos [batch, seq_len, vocab_size] para compatibilidad T5
            logits = logits.transpose(0, 1)
            
            # TODO: Extraer past_key_values del decoder
            # Por ahora, devolver None para compatibilidad
            past_key_values_out = None
            
        else:
            # Pasos siguientes: usar cache
            logger.debug("  -> Paso con cache")
            
            # Por ahora, usar el decoder sin cache como fallback
            logits = self.decoder.forward(
                frame_embs=frame_embs,
                frame_embs_pad_mask=frame_embs_pad_mask,
                caps_in=caps_in,
                caps_in_pad_mask=caps_in_pad_mask,
                caps_in_sq_mask=caps_in_sq_mask,
            )
            
            logits = logits.transpose(0, 1)
            past_key_values_out = past_key_values  # Mantener el mismo cache
        
        return {
            'last_hidden_state': logits,
            'past_key_values': past_key_values_out
        }

class CoNeTTET5Approach:
    """
    Implementación del enfoque T5 para CoNeTTE usando exportación funcional.
    """
    
    def __init__(self):
        self.setup_models()
        self.prepare_paths()
    
    def setup_models(self):
        """Configurar modelos PyTorch."""
        logger.info("🔧 Configurando modelos CoNeTTE...")
        
        # Cargar modelo PyTorch
        try:
            from conette import CoNeTTEConfig, CoNeTTEModel
            
            config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
            self.pytorch_model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
            self.pytorch_model = self.pytorch_model.float().eval()
            
            # Crear wrapper exportable
            self.export_model = ExportCoNeTTE(self.pytorch_model).eval()
            
            logger.info("✅ Modelos configurados")
            
        except Exception as e:
            logger.error(f"❌ Error configurando modelos: {e}")
            raise
    
    def prepare_paths(self):
        """Preparar carpetas para modelos ONNX."""
        
        def prepare_folder(path: str) -> str:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            # Limpiar archivos existentes si se solicita
            for item in p.glob("model.onnx"):
                if item.is_file():
                    logger.info(f"Removiendo modelo existente: {item}")
                    item.unlink()
            return path + "/model.onnx"
        
        self.dec_no_cache_path = prepare_folder(str(T5_MODELS_DIR))
        
        logger.info("📁 Carpetas preparadas")
    
    def get_sample_inputs(self):
        """Obtener inputs de muestra para export usando enfoque funcional."""
        
        # Crear audio de ejemplo
        audio = torch.randn(1, 32000 * 5, dtype=torch.float32)  # 5 segundos a 32kHz
        
        # Obtener features reales del encoder usando el modelo completo
        with torch.no_grad():
            # Usar el preprocessor exactamente como en el modelo
            batch = self.pytorch_model.preprocessor(audio)
            batch["dataset"] = ["clotho"]
            batch["source"] = [None]
            
            # Obtener encoder outputs
            encoder_outputs = self.pytorch_model.model.encoder(batch["audio"], batch["audio_shape"])
            frame_embs_raw = encoder_outputs["frame_embs"]
            
            # Aplicar projection
            frame_embs_projected = self.pytorch_model.model.projection(frame_embs_raw)
            
            # Transponer de [batch, d_model, time] a [batch, time, d_model] 
            encoder_features = frame_embs_projected.transpose(1, 2)
        
        # Input IDs de muestra
        input_ids = torch.tensor([[self.export_model.bos_id]], dtype=torch.long)
        
        logger.info(f"Sample inputs preparados:")
        logger.info(f"  input_ids: {input_ids.shape}")
        logger.info(f"  encoder_features: {encoder_features.shape}")
        
        return input_ids, encoder_features
    
    def export_no_cache_model(self):
        """Exportar modelo sin cache (para primer token) usando implementación funcional."""
        logger.info("📦 Exportando modelo SIN cache...")
        
        input_ids, encoder_features = self.get_sample_inputs()
        
        # Preparar inputs para export
        model_inputs = (input_ids, encoder_features)
        
        # Definir nombres de inputs/outputs
        input_names = ["input_ids", "encoder_hidden_states"]
        output_names = ["logits"]  # Sin past_key_values por ahora
        
        # Definir ejes dinámicos
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
            "logits": {0: "batch", 1: "sequence"},
        }
        
        # Exportar modelo
        try:
            with torch.no_grad():
                torch.onnx.export(
                    self.export_model,
                    model_inputs,
                    f=self.dec_no_cache_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    opset_version=14,  # Usar opset 14 para soportar torch.triu
                    verbose=False  # Reducir verbosidad
                )
            
            logger.info(f"✅ Modelo sin cache exportado: {self.dec_no_cache_path}")
            return self.dec_no_cache_path
            
        except Exception as e:
            logger.error(f"❌ Error exportando modelo: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_exported_models(self):
        """Probar modelos exportados vs PyTorch."""
        logger.info("🧪 Probando modelos exportados...")
        
        input_ids, encoder_features = self.get_sample_inputs()
        
        # 1. Resultado PyTorch
        with torch.no_grad():
            pytorch_result = self.export_model(
                input_ids=input_ids,
                encoder_hidden_states=encoder_features,
                past_key_values=None
            )
        
        pytorch_logits = pytorch_result['last_hidden_state']
        logger.info(f"PyTorch logits: {pytorch_logits.shape}")
        
        # 2. Resultado ONNX sin cache
        try:
            session = ort.InferenceSession(self.dec_no_cache_path)
            
            onnx_inputs = {
                'input_ids': input_ids.numpy().astype(np.int64),
                'encoder_hidden_states': encoder_features.numpy().astype(np.float32)
            }
            
            onnx_outputs = session.run(None, onnx_inputs)
            onnx_logits = torch.from_numpy(onnx_outputs[0])
            
            logger.info(f"ONNX logits: {onnx_logits.shape}")
            
            # Comparar resultados
            diff = torch.abs(pytorch_logits - onnx_logits).max().item()
            logger.info(f"Diferencia máxima PyTorch vs ONNX: {diff:.6f}")
            
            if diff < 1e-3:
                logger.info("✅ Modelos coinciden!")
                return True
            else:
                logger.warning(f"⚠️ Diferencias detectadas: {diff}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error probando ONNX: {e}")
            return False
    
    def create_symbolic_links(self):
        """Crear enlaces simbólicos para mantener compatibilidad con rutas funcionales."""
        logger.info("🔗 Creando enlaces simbólicos para compatibilidad...")
        
        # Rutas de origen y destino
        t5_src = Path(self.dec_no_cache_path)
        t5_dst = T5_FUNCTIONAL_DIR / "model.onnx"
        
        # Crear enlace simbólico si el archivo fuente existe
        try:
            if t5_src.exists():
                # Remover enlace existente si existe
                if t5_dst.exists() or t5_dst.is_symlink():
                    t5_dst.unlink()
                
                # Crear enlace simbólico relativo
                rel_path = os.path.relpath(t5_src, t5_dst.parent)
                t5_dst.symlink_to(rel_path)
                logger.info(f"✅ Enlace simbólico creado: {t5_dst} -> {rel_path}")
                
        except Exception as e:
            logger.error(f"❌ Error creando enlace simbólico: {e}")
    
    def run_export_process(self):
        """Ejecutar proceso de exportación completo."""
        logger.info("🚀 INICIANDO EXPORTACIÓN T5 PARA CONETTE")
        logger.info("=" * 60)
        
        try:
            # 1. Exportar modelo
            self.export_no_cache_model()
            
            # 2. Probar modelo
            test_success = self.test_exported_models()
            
            # 3. Crear enlaces simbólicos
            self.create_symbolic_links()
            
            # 4. Verificar modelo ONNX
            verification_success = self.verify_onnx_model()
            
            # 5. Resultados finales
            logger.info("\\n=== RESULTADOS FINALES ===")
            if test_success and verification_success:
                logger.info("✅ EXPORTACIÓN T5 COMPLETADA EXITOSAMENTE")
                logger.info(f"📁 Modelo guardado en: {self.dec_no_cache_path}")
                logger.info(f"🔗 Enlace funcional en: {T5_FUNCTIONAL_DIR / 'model.onnx'}")
                return True
            else:
                logger.warning("⚠️ Exportación completada con advertencias")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error en exportación T5: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def verify_onnx_model(self):
        """Verificar que el modelo ONNX exportado funcione correctamente."""
        logger.info("🔍 Verificando modelo ONNX...")
        
        try:
            import onnx
            import onnxruntime as ort
            
            model_path = Path(self.dec_no_cache_path)
            
            if model_path.exists():
                logger.info(f"Verificando {model_path.name}...")
                
                try:
                    # Cargar y verificar el modelo con onnx
                    onnx_model = onnx.load(model_path.as_posix())
                    onnx.checker.check_model(onnx_model)
                    
                    # Crear sesión de ONNX Runtime
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    sess = ort.InferenceSession(
                        model_path.as_posix(),
                        sess_options,
                        providers=["CPUExecutionProvider"]
                    )
                    
                    logger.info(f"  ✅ {model_path.name} verificado correctamente")
                    logger.info(f"  📊 Entradas: {[i.name for i in sess.get_inputs()]}")
                    logger.info(f"  📊 Salidas: {[o.name for o in sess.get_outputs()]}")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"  ❌ Error verificando {model_path.name}: {e}")
                    return False
            else:
                logger.error(f"  ❌ No se encontró el archivo: {model_path}")
                return False
                
        except ImportError:
            logger.warning("⚠️ onnx o onnxruntime no disponibles para verificación")
            return True
        except Exception as e:
            logger.error(f"❌ Error durante verificación: {e}")
            return False

def main():
    """
    Función principal que coordina el proceso de exportación T5.
    """
    logger.info("\\n" + "="*80)
    logger.info(" EXPORTACIÓN T5 MODELS CONETTE A ONNX - VERSIÓN FUNCIONAL ")
    logger.info("="*80 + "\\n")
    
    try:
        approach = CoNeTTET5Approach()
        success = approach.run_export_process()
        
        if success:
            logger.info("\\n🎉 PROCESO T5 COMPLETADO EXITOSAMENTE")
            logger.info("El modelo T5 está listo para usar con sistemas de inferencia")
        else:
            logger.error("\\n❌ PROCESO T5 FALLÓ")
            logger.info("Revise los mensajes anteriores para detalles")
        
        return success
        
    except Exception as e:
        logger.error(f"\\n❌ ERROR CRÍTICO: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error no controlado: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)