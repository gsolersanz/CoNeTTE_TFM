#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TEST TOKENIZER STANDALONE
=========================

Prueba el tokenizer extraído para verificar que funciona correctamente.
"""

import sys
import time
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_standalone_tokenizer():
    """Probar tokenizer standalone."""
    
    logger.info("🧪 Probando tokenizer standalone...")
    
    try:
        # Importar cargador
        sys.path.append('../06_models/conette_tokenizer_standalone')
        from load_tokenizer import StandaloneTokenizerLoader
        
        # Cargar tokenizer
        start_time = time.time()
        loader = StandaloneTokenizerLoader('../06_models/conette_tokenizer_standalone')
        tokenizer, metadata = loader.load()
        load_time = time.time() - start_time
        
        logger.info(f"✅ Tokenizer cargado en {load_time:.3f}s")
        logger.info(f"📊 Vocab size: {metadata['vocab_size']}")
        
        # Test encode/decode
        test_texts = [
            "a woman is singing",
            "rain is falling",
            "bicycle bell ringing",
            "children playing in background"
        ]
        
        for text in test_texts:
            # Encode
            tokens = tokenizer.encode_single(text)
            
            # Decode
            if hasattr(tokenizer, 'decode_batch'):
                decoded = tokenizer.decode_batch([tokens])[0]
            elif hasattr(tokenizer, 'decode_single'):
                decoded = tokenizer.decode_single(tokens)
            else:
                decoded = "Manual decode needed"
            
            logger.info(f"'{text}' → {tokens[:5]}... → '{decoded}'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_standalone_tokenizer()