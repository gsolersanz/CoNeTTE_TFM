#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARGADOR RÁPIDO DE TOKENIZER STANDALONE
======================================

Carga el tokenizer extraído sin dependencias de PyTorch/HuggingFace.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

class StandaloneTokenizerLoader:
    """Cargador optimizado para tokenizer standalone."""
    
    def __init__(self, tokenizer_dir: str = "."):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer_path = self.tokenizer_dir / "tokenizer.pkl"
        self.metadata_path = self.tokenizer_dir / "metadata.json"
    
    def load(self) -> Tuple[Any, Dict[str, Any]]:
        """Cargar tokenizer y metadatos."""
        
        # Cargar tokenizer
        with open(self.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Cargar metadatos
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return tokenizer, metadata
    
    def get_special_tokens(self) -> Dict[str, Any]:
        """Obtener tokens especiales sin cargar el tokenizer completo."""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata

if __name__ == "__main__":
    # Test del cargador
    loader = StandaloneTokenizerLoader()
    tokenizer, metadata = loader.load()
    
    print(f"✅ Tokenizer cargado: {metadata['cls']}")
    print(f"📊 Vocab size: {metadata['vocab_size']}")
    print(f"🎯 BOS: {metadata['bos_token_id']}, EOS: {metadata['eos_token_id']}")
    
    # Test de funcionalidad
    test_tokens = [metadata['bos_token_id'], 4, 175, metadata['eos_token_id']]
    if metadata['has_decode_batch']:
        decoded = tokenizer.decode_batch([test_tokens])[0]
        print(f"✅ Test decode: {test_tokens} → '{decoded}'")
