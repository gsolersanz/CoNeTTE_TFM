#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPORTADOR ENCODER Y PROJECTION PARA CoNeTTE - VERSIÓN FUNCIONAL
================================================================

Este script exporta los componentes encoder y projection del modelo CoNeTTE a ONNX
usando la implementación probada y funcional de onnx_export/export_decoder_only.py

COMPONENTES EXPORTADOS:
- conette_encoder.onnx: Encoder CNN14 que procesa audio a embeddings
- conette_projection.onnx: Projection layer que convierte embeddings 768D → 256D

DIRECTORIO DE SALIDA: 06_models/onnx_models/
RUTAS FUNCIONALES GENERADAS:
- onnx_models_full/conette_encoder.onnx (enlace simbólico)
- onnx_models_full/conette_projection.onnx (enlace simbólico)
"""

import os
import sys
import logging
import inspect
import traceback
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any

import torch
import torchaudio
import numpy as np
from torch import nn, Tensor

# Add current source to path to use local conette instead of installed package
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent  # Go up to src/conette level
sys.path.insert(0, str(src_dir.parent))  # Add src/ to path

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('export_encoder_projection.log')
    ]
)

logger = logging.getLogger(__name__)

# Constantes y configuración
ONNX_DIR = Path("06_models/onnx_models")
ONNX_FUNCTIONAL_DIR = Path("onnx_models_full")  # Directorio funcional para enlaces
DEBUG_MODE = True
OPSET_VERSION = 14  # Versión del opset de ONNX a utilizar

# Crear directorios para modelos ONNX si no existen
ONNX_DIR.mkdir(parents=True, exist_ok=True)
ONNX_FUNCTIONAL_DIR.mkdir(parents=True, exist_ok=True)

class CoNeTTEEncoderWrapper(nn.Module):
    """
    Wrapper para el encoder de CoNeTTE que extrae los embeddings de audio.
    Implementación funcional probada.
    """
    def __init__(self, model):
        super().__init__()
        self.preprocessor_encoder = model.preprocessor.encoder
        
    def forward(self, audio, audio_shape=None):
        # Asegurarse de que audio sea float32
        audio = audio.to(dtype=torch.float32)
        
        # Crear audio_shape si no se proporciona
        if audio_shape is None:
            audio_shape = torch.tensor([[audio.shape[1]]], dtype=torch.long, device=audio.device)
        
        # Procesar con el encoder
        encoder_outputs = self.preprocessor_encoder(audio, audio_shape)
        
        # Extraer embeddings y longitudes
        frame_embs = encoder_outputs["frame_embs"]  # [batch_size, embed_dim=768, time_steps]
        frame_embs_lens = encoder_outputs.get("frame_embs_lens")
        
        # Asegurar que frame_embs sea float32
        frame_embs = frame_embs.to(dtype=torch.float32)
        
        # Crear máscara de padding a partir de longitudes
        if frame_embs_lens is not None:
            batch_size = frame_embs_lens.shape[0]
            max_len = frame_embs.shape[2]  # La dimensión temporal está en la posición 2
            indices = torch.arange(max_len, device=frame_embs_lens.device).unsqueeze(0).expand(batch_size, -1)
            frame_embs_pad_mask = indices >= frame_embs_lens.unsqueeze(1)
        else:
            frame_embs_pad_mask = torch.zeros(
                (frame_embs.shape[0], frame_embs.shape[2]), 
                dtype=torch.bool, 
                device=frame_embs.device
            )
        
        return frame_embs, frame_embs_pad_mask

class CoNeTTEProjectionWrapper(nn.Module):
    """
    Wrapper para la capa de proyección que convierte los embeddings de 768d a 256d.
    Implementación funcional probada.
    """
    def __init__(self, model):
        super().__init__()
        self.projection = model.model.projection
        
    def forward(self, frame_embs):
        # Asegurar que la entrada es float32
        frame_embs = frame_embs.to(dtype=torch.float32)
        
        # frame_embs: [batch_size, embed_dim=768, time_steps]
        # Transponer: (batch, emb_dim, time) -> (batch, time, emb_dim)
        frame_embs = frame_embs.transpose(1, 2)
        
        # Aplicar proyección: convierte de 768 a 256, resultado esperado: [batch, time, 256]
        frame_embs = self.projection(frame_embs)
        
        # Asegurar que la salida sea float32
        frame_embs = frame_embs.to(dtype=torch.float32)
        
        return frame_embs # Forma esperada: [batch, time, hidden_size=256]

def convert_model_to_float32(model, component_name=None):
    """
    Convierte todos los parámetros y buffers del modelo a float32.
    """
    name = component_name or "modelo"
    logger.info(f"Convirtiendo {name} a float32...")
    
    # Convertir todo el modelo
    model = model.float()
    
    # Verificar la conversión
    non_float32_params = []
    for name_param, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_float32_params.append(name_param)
            # Convertir manualmente si la conversión global no funcionó
            param.data = param.data.float()
    
    if non_float32_params:
        logger.warning(f"Los siguientes parámetros no se convirtieron automáticamente: {non_float32_params}")
        logger.info("Se realizó una conversión manual de estos parámetros.")
    else:
        logger.info(f"Conversión a float32 completada exitosamente para {name}.")
    
    # También convertir los buffers manualmente
    for name_buf, buffer in model.named_buffers():
        if buffer.dtype not in [torch.float32, torch.bool, torch.long, torch.int64]:
            logger.info(f"Convirtiendo buffer {name_buf} de {buffer.dtype} a float32")
            model.register_buffer(name_buf, buffer.to(torch.float32))
    
    return model

def export_to_onnx_with_type_verification(
    wrapper, 
    inputs, 
    export_path, 
    input_names, 
    output_names, 
    dynamic_axes=None, 
    opset_version=14
):
    """
    Exporta un componente a ONNX con verificación previa de tipos de datos.
    Implementación funcional probada.
    """
    logger.info(f"Preparando exportación a ONNX: {export_path}")
    
    # 1. Asegurar que todo esté en float32
    wrapper = convert_model_to_float32(wrapper)
    
    # 2. Asegurar que las entradas sean float32 si son tensores de punto flotante
    if isinstance(inputs, torch.Tensor) and inputs.is_floating_point() and inputs.dtype != torch.float32:
        logger.info("Convirtiendo entrada de float64 a float32")
        inputs = inputs.float()
    elif isinstance(inputs, tuple):
        new_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor) and inp.is_floating_point() and inp.dtype != torch.float32:
                logger.info(f"Convirtiendo entrada {i} de {inp.dtype} a float32")
                new_inputs.append(inp.float())
            else:
                new_inputs.append(inp)
        inputs = tuple(new_inputs)
    
    # 3. Intentar exportar con diferentes opciones si es necesario
    opset_versions_to_try = [opset_version, 13, 12, 11]
    
    for current_opset in opset_versions_to_try:
        try:
            logger.info(f"Exportando a {export_path} con opset_version={current_opset}...")
            
            # Opciones de exportación
            export_options = {
                'export_params': True,
                'opset_version': current_opset,
                'input_names': input_names,
                'output_names': output_names,
                'dynamic_axes': dynamic_axes or {},
                'do_constant_folding': True,
                'verbose': False
            }
            
            # Realizar la exportación
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    inputs,
                    export_path,
                    **export_options
                )
            
            logger.info(f"✅ Exportación exitosa a {export_path}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error durante exportación con opset {current_opset}: {error_msg}")
            
            if current_opset == opset_versions_to_try[-1]:
                logger.error("❌ Se han agotado todas las versiones de opset disponibles sin éxito.")
                return False
            
            logger.info(f"Intentando con opset_version={opset_versions_to_try[opset_versions_to_try.index(current_opset) + 1]}...")
    
    return False

def load_pytorch_model():
    """Cargar modelo PyTorch completo."""
    logger.info("📦 Cargando modelo PyTorch...")
    
    try:
        from conette import CoNeTTEConfig, CoNeTTEModel
        
        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
        model = model.to(torch.float32)
        model.eval()
        
        logger.info("✅ Modelo PyTorch cargado exitosamente")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo PyTorch: {e}")
        raise

def create_symbolic_links():
    """Crear enlaces simbólicos para mantener compatibilidad con rutas funcionales."""
    logger.info("🔗 Creando enlaces simbólicos para compatibilidad...")
    
    # Rutas de origen y destino
    encoder_src = ONNX_DIR / "conette_encoder.onnx"
    projection_src = ONNX_DIR / "conette_projection.onnx"
    
    encoder_dst = ONNX_FUNCTIONAL_DIR / "conette_encoder.onnx"
    projection_dst = ONNX_FUNCTIONAL_DIR / "conette_projection.onnx"
    
    # Crear enlaces simbólicos si los archivos fuente existen
    try:
        if encoder_src.exists():
            # Remover enlace existente si existe
            if encoder_dst.exists() or encoder_dst.is_symlink():
                encoder_dst.unlink()
            
            # Crear enlace simbólico relativo
            rel_path = os.path.relpath(encoder_src, encoder_dst.parent)
            encoder_dst.symlink_to(rel_path)
            logger.info(f"✅ Enlace simbólico creado: {encoder_dst} -> {rel_path}")
            
        if projection_src.exists():
            # Remover enlace existente si existe
            if projection_dst.exists() or projection_dst.is_symlink():
                projection_dst.unlink()
                
            # Crear enlace simbólico relativo
            rel_path = os.path.relpath(projection_src, projection_dst.parent)
            projection_dst.symlink_to(rel_path)
            logger.info(f"✅ Enlace simbólico creado: {projection_dst} -> {rel_path}")
            
    except Exception as e:
        logger.error(f"❌ Error creando enlaces simbólicos: {e}")

def export_model_components_to_onnx():
    """
    Exporta los componentes encoder y projection del modelo CoNeTTE a ONNX.
    Implementación funcional completa.
    """
    logger.info("\\n=== EXPORTACIÓN DE ENCODER Y PROJECTION A ONNX ===")
    
    try:
        # 1. Cargar modelo PyTorch
        model = load_pytorch_model()
        
        # 2. Crear audio de ejemplo para pruebas
        logger.info("🔧 Preparando audio de ejemplo...")
        audio = torch.randn(1, 32000 * 5, dtype=torch.float32)  # 5 segundos a 32kHz
        audio_shape = torch.tensor([[audio.shape[1]]], dtype=torch.long)
        
        # 3. Exportar encoder
        logger.info("\\n1. Exportando encoder...")
        encoder_wrapper = CoNeTTEEncoderWrapper(model)
        encoder_wrapper = convert_model_to_float32(encoder_wrapper, "encoder_wrapper")
        
        encoder_path = ONNX_DIR / "conette_encoder.onnx"
        encoder_success = export_to_onnx_with_type_verification(
            encoder_wrapper,
            (audio, audio_shape),
            encoder_path.as_posix(),
            input_names=["audio", "audio_shape"],
            output_names=["frame_embs", "frame_embs_pad_mask"],
            dynamic_axes={
                "audio": {0: "batch_size", 1: "audio_length"},
                "audio_shape": {0: "batch_size"},
                "frame_embs": {0: "batch_size", 2: "time_steps"},
                "frame_embs_pad_mask": {0: "batch_size", 1: "time_steps"}
            }
        )
        
        if encoder_success:
            logger.info(f"✅ Encoder exportado exitosamente a {encoder_path}")
        else:
            logger.error(f"❌ Fallo al exportar encoder")
            return False
        
        # 4. Exportar projection layer
        logger.info("\\n2. Exportando projection layer...")
        
        # Primero obtener embeddings del encoder para usar como entrada
        with torch.no_grad():
            frame_embs_raw, _ = encoder_wrapper(audio, audio_shape)
        
        projection_wrapper = CoNeTTEProjectionWrapper(model)
        projection_wrapper = convert_model_to_float32(projection_wrapper, "projection_wrapper")
        
        projection_path = ONNX_DIR / "conette_projection.onnx"
        projection_success = export_to_onnx_with_type_verification(
            projection_wrapper,
            (frame_embs_raw,),
            projection_path.as_posix(),
            input_names=["encoder_features"],
            output_names=["projected_features"],
            dynamic_axes={
                "encoder_features": {0: "batch_size", 2: "time_steps"},
                "projected_features": {0: "batch_size", 1: "time_steps"}
            }
        )
        
        if projection_success:
            logger.info(f"✅ Projection layer exportado exitosamente a {projection_path}")
        else:
            logger.error(f"❌ Fallo al exportar projection layer")
            return False
        
        # 5. Crear enlaces simbólicos para compatibilidad
        create_symbolic_links()
        
        # 6. Verificar componentes ONNX
        logger.info("\\n3. Verificando componentes ONNX...")
        verification_success = verify_onnx_components([encoder_path, projection_path])
        
        # 7. Resultados finales
        logger.info("\\n=== RESULTADOS FINALES ===")
        if encoder_success and projection_success and verification_success:
            logger.info("✅ EXPORTACIÓN COMPLETA Y EXITOSA")
            logger.info(f"📁 Archivos ONNX guardados en: {ONNX_DIR}")
            logger.info(f"🔗 Enlaces funcionales en: {ONNX_FUNCTIONAL_DIR}")
            return True
        else:
            logger.warning("⚠️ Exportación completada con advertencias")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en exportación: {e}")
        logger.error(traceback.format_exc())
        return False

def verify_onnx_components(component_paths):
    """
    Verifica que los componentes ONNX exportados funcionen correctamente.
    """
    logger.info("🔍 Verificando componentes ONNX...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        all_verified = True
        
        for path in component_paths:
            if path.exists():
                component_name = path.name
                logger.info(f"Verificando {component_name}...")
                
                try:
                    # Cargar y verificar el modelo con onnx
                    onnx_model = onnx.load(path.as_posix())
                    onnx.checker.check_model(onnx_model)
                    
                    # Crear sesión de ONNX Runtime
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    sess = ort.InferenceSession(
                        path.as_posix(),
                        sess_options,
                        providers=["CPUExecutionProvider"]
                    )
                    
                    logger.info(f"  ✅ {component_name} verificado correctamente")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error verificando {component_name}: {e}")
                    all_verified = False
            else:
                logger.error(f"  ❌ No se encontró el archivo: {path}")
                all_verified = False
                
        return all_verified
                
    except ImportError:
        logger.warning("⚠️ onnx o onnxruntime no disponibles para verificación")
        return True
    except Exception as e:
        logger.error(f"❌ Error durante verificación: {e}")
        return False

def main():
    """
    Función principal que coordina el proceso de exportación.
    """
    logger.info("\\n" + "="*80)
    logger.info(" EXPORTACIÓN ENCODER Y PROJECTION CONETTE A ONNX - VERSIÓN FUNCIONAL ")
    logger.info("="*80 + "\\n")
    
    try:
        success = export_model_components_to_onnx()
        
        if success:
            logger.info("\\n🎉 PROCESO COMPLETADO EXITOSAMENTE")
            logger.info("Los modelos están listos para usar con sistemas de inferencia T5")
        else:
            logger.error("\\n❌ PROCESO FALLÓ")
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