#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WORKFLOW COMPLETO DEFINITIVO
============================

Ejecuta todo el pipeline en orden:
1. Exportar encoder y projection
2. Exportar T5 models  
3. Extraer tokenizer standalone
4. Ejecutar inferencia de prueba

Este script debe ejecutarse desde el directorio Definitivo/
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_step(script_path: str, description: str) -> bool:
    """Ejecutar un paso del workflow."""
    
    logger.info(f"🔄 {description}")
    logger.info(f"   Ejecutando: {script_path}")
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - EXITOSO")
            if result.stdout:
                # Mostrar últimas líneas del output
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:  # Últimas 3 líneas
                    if line.strip():
                        logger.info(f"   {line}")
            return True
        else:
            logger.error(f"❌ {description} - FALLÓ")
            if result.stderr:
                logger.error(f"   Error: {result.stderr}")
            if result.stdout:
                logger.error(f"   Output: {result.stdout}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error ejecutando {script_path}: {e}")
        return False

def check_models_exist() -> dict:
    """Verificar qué modelos ya existen."""
    
    models_dir = Path("06_models")
    functional_dirs = {
        "onnx_models_full": Path("onnx_models_full"),
        "conette_t5": Path("conette_t5")
    }
    
    checks = {
        "encoder": models_dir / "onnx_models" / "conette_encoder.onnx",
        "projection": models_dir / "onnx_models" / "conette_projection.onnx", 
        "t5_decoder": models_dir / "t5_models" / "dec_no_cache" / "model.onnx",
        "tokenizer": models_dir / "conette_tokenizer_standalone" / "tokenizer.pkl",
        # Enlaces funcionales
        "encoder_link": functional_dirs["onnx_models_full"] / "conette_encoder.onnx",
        "projection_link": functional_dirs["onnx_models_full"] / "conette_projection.onnx",
        "t5_decoder_link": functional_dirs["conette_t5"] / "dec_no_cache" / "model.onnx"
    }
    
    status = {}
    for name, path in checks.items():
        exists = path.exists()
        status[name] = exists
        symbol = "✅" if exists else "❌"
        # Mostrar si es enlace simbólico
        link_info = " (link)" if path.is_symlink() else ""
        logger.info(f"   {symbol} {name}: {path}{link_info}")
    
    return status

def main():
    """Workflow principal."""
    
    logger.info("🚀 WORKFLOW COMPLETO DEFINITIVO")
    logger.info("=" * 50)
    
    # Verificar directorio actual
    if not Path("01_export_pipeline").exists():
        logger.error("❌ Error: Debe ejecutarse desde directorio Definitivo/")
        logger.info("💡 cd a /workspace/conette/Definitivo/")
        return False
    
    # 1. Verificar estado actual
    logger.info("📋 Verificando modelos existentes...")
    status = check_models_exist()
    
    # Verificar modelos principales (no enlaces)
    main_models = ["encoder", "projection", "t5_decoder", "tokenizer"]
    missing_models = [name for name in main_models if not status.get(name, False)]
    
    if not missing_models:
        logger.info("✅ Todos los modelos principales existen!")
        # Verificar enlaces funcionales
        functional_links = ["encoder_link", "projection_link", "t5_decoder_link"]
        missing_links = [name for name in functional_links if not status.get(name, False)]
        
        if missing_links:
            logger.info("🔗 Recreando enlaces funcionales...")
            # Re-ejecutar exports para crear enlaces
            run_step("01_export_pipeline/export_encoder_projection.py", "Recrear enlaces encoder/projection")
            run_step("01_export_pipeline/export_t5_models.py", "Recrear enlaces T5")
        
        logger.info("🎯 Ejecutando test de inferencia...")
        return run_step("04_inference_systems/t5_onnx_inference.py", "Test de inferencia completo")
    
    logger.info(f"📝 Modelos faltantes: {missing_models}")
    
    # 2. Exportar encoder y projection si faltan
    if "encoder" in missing_models or "projection" in missing_models:
        success = run_step(
            "01_export_pipeline/export_encoder_projection.py",
            "Exportar encoder y projection (versión funcional)"
        )
        if not success:
            logger.error("❌ Falló exportación de encoder/projection")
            return False
    
    # 3. Exportar T5 models si faltan
    if "t5_decoder" in missing_models:
        success = run_step(
            "01_export_pipeline/export_t5_models.py", 
            "Exportar T5 decoder models (versión funcional)"
        )
        if not success:
            logger.error("❌ Falló exportación de T5 decoder")
            return False
    
    # 4. Extraer tokenizer si falta
    if "tokenizer" in missing_models:
        success = run_step(
            "02_tokenizer_extraction/extract_standalone_tokenizer.py",
            "Extraer tokenizer standalone"
        )
        if not success:
            logger.error("❌ Falló extracción de tokenizer")
            return False
    
    # 5. Test de tokenizer
    logger.info("🧪 Probando tokenizer extraído...")
    run_step(
        "02_tokenizer_extraction/test_tokenizer.py",
        "Test de tokenizer standalone"
    )
    
    # 6. Verificar que todo está listo
    logger.info("📋 Verificación final...")
    final_status = check_models_exist()
    
    all_ready = all(final_status.values())
    if not all_ready:
        logger.error("❌ Algunos modelos siguen faltando")
        return False
    
    # 7. Test de inferencia completo
    logger.info("🎯 Ejecutando test de inferencia completo...")
    success = run_step(
        "04_inference_systems/t5_onnx_inference.py",
        "Test de inferencia completo"
    )
    
    if success:
        logger.info("\n🎉 WORKFLOW COMPLETADO EXITOSAMENTE")
        logger.info("✅ Todos los componentes funcionando")
        logger.info("🚀 Sistema listo para producción")
    else:
        logger.error("\n❌ WORKFLOW FALLÓ EN TEST FINAL")
        logger.info("💡 Revisa los logs arriba para detalles")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)