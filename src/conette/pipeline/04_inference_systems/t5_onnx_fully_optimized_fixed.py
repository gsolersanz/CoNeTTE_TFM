#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T5 ONNX INFERENCE COMPLETAMENTE OPTIMIZADO - VERSION FUNCIONAL
==============================================================
"""

import os
import sys
import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar tokenizer standalone con la ruta correcta
script_dir = Path(__file__).parent
base_dir = script_dir.parent.parent
tokenizer_path = base_dir / "Definitivo/06_models/conette_tokenizer_standalone"

STANDALONE_TOKENIZER_AVAILABLE = False
if tokenizer_path.exists():
    sys.path.append(str(tokenizer_path))
    try:
        from load_tokenizer import StandaloneTokenizerLoader
        STANDALONE_TOKENIZER_AVAILABLE = True
        logger.info(f"✅ Tokenizer standalone disponible en: {tokenizer_path}")
    except ImportError:
        logger.warning("⚠️ No se pudo importar StandaloneTokenizerLoader")
else:
    logger.warning(f"⚠️ Tokenizer standalone no encontrado en: {tokenizer_path}")

class T5ONNXFullyOptimized:
    """Sistema de inferencia ONNX con todas las optimizaciones."""
    
    def __init__(self,
                 decoder_path: str = "Definitivo/06_models/t5_models/dec_no_cache/model.onnx",
                 encoder_path: str = "Definitivo/06_models/onnx_models/conette_encoder.onnx", 
                 projection_path: str = "Definitivo/06_models/onnx_models/conette_projection.onnx",
                 tokenizer_dir: str = "Definitivo/06_models/conette_tokenizer_standalone",
                 enable_zero_copy: bool = True,
                 enable_fp16: bool = True,
                 enable_tensorrt: bool = True,
                 jetson_nano_mode: bool = True):
        
        logger.info("🚀 Inicializando T5ONNXFullyOptimized")
        
        self.enable_zero_copy = enable_zero_copy
        self.enable_fp16 = enable_fp16
        self.enable_tensorrt = enable_tensorrt
        self.jetson_nano_mode = jetson_nano_mode
        
        # Cargar tokenizer
        self._load_tokenizer_optimized(tokenizer_dir)
        
        # Configurar providers
        self.providers = self._setup_providers()
        
        # Cargar modelos
        self._load_models(decoder_path, encoder_path, projection_path)
        
        # Configurar zero-copy si está habilitado
        if self.enable_zero_copy:
            self._setup_zero_copy_system()
        
        # Cargar preprocessor
        self._load_preprocessor()
        
        # Parámetros de generación
        self.beam_size = 3
        self.max_length = 20
        self.min_pred_size = 1
        
        logger.info("✅ Sistema inicializado correctamente")
    
    def _load_tokenizer_optimized(self, tokenizer_dir: str):
        """Cargar tokenizer standalone o fallback."""
        base_dir = Path(__file__).parent.parent.parent
        tokenizer_full_path = base_dir / tokenizer_dir
        
        if STANDALONE_TOKENIZER_AVAILABLE and tokenizer_full_path.exists():
            try:
                logger.info("⚡ Cargando tokenizer standalone...")
                self.tokenizer_loader = StandaloneTokenizerLoader(str(tokenizer_full_path))
                self.tokenizer, self.metadata = self.tokenizer_loader.load()
                
                self.special_tokens = self.metadata.copy()
                if 'special_tokens' in self.metadata and '<bos_clotho>' in self.metadata['special_tokens']:
                    self.special_tokens['clotho_bos_id'] = self.metadata['special_tokens']['<bos_clotho>']
                else:
                    self.special_tokens['clotho_bos_id'] = self.metadata.get('bos_token_id', 1)
                
                logger.info("✅ Tokenizer standalone cargado")
                return
            except Exception as e:
                logger.error(f"Error cargando tokenizer standalone: {e}")
        
        # Fallback al tokenizer completo
        logger.info("📦 Usando tokenizer completo de HuggingFace...")
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))
        
        from conette.huggingface.model import CoNeTTEModel
        from conette.huggingface.config import CoNeTTEConfig
        
        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        conette_model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
        
        self.tokenizer = conette_model.model.tokenizer
        self.special_tokens = {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'clotho_bos_id': self.tokenizer.token_to_id('<bos_clotho>') if hasattr(self.tokenizer, 'token_to_id') else self.tokenizer.bos_token_id
        }
    
    def _setup_providers(self) -> List:
        """Configurar providers ONNX."""
        providers = []
        available = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available:
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 1024 * 1024 * 512 if self.jetson_nano_mode else 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True
            }
            providers.append(('CUDAExecutionProvider', cuda_options))
            logger.info("   ✅ CUDA Provider configurado")
        
        providers.append('CPUExecutionProvider')
        logger.info("   ✅ CPU Provider configurado")
        
        return providers
    
    def _load_models(self, decoder_path: str, encoder_path: str, projection_path: str):
        """Cargar modelos ONNX."""
        base_dir = Path(__file__).parent.parent.parent
        decoder_path = str(base_dir / decoder_path)
        encoder_path = str(base_dir / encoder_path) 
        projection_path = str(base_dir / projection_path)
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        logger.info("📦 Cargando modelos ONNX...")
        self.decoder_session = ort.InferenceSession(decoder_path, providers=self.providers, sess_options=session_options)
        self.encoder_session = ort.InferenceSession(encoder_path, providers=self.providers, sess_options=session_options)
        self.projection_session = ort.InferenceSession(projection_path, providers=self.providers, sess_options=session_options)
        logger.info("✅ Modelos cargados")
    
    def _setup_zero_copy_system(self):
        """Configurar zero-copy."""
        self.device_type = 'cpu'
        self.device_id = 0
        
        self.encoder_io_binding = self.encoder_session.io_binding()
        self.projection_io_binding = self.projection_session.io_binding()
        self.decoder_io_binding = self.decoder_session.io_binding()
        
        logger.info("✅ Zero-copy configurado")
    
    def _load_preprocessor(self):
        """Cargar preprocessor."""
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))
        
        from conette.huggingface.model import CoNeTTEModel
        from conette.huggingface.config import CoNeTTEConfig
        
        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
        self._preprocessor = model.preprocessor
        logger.info("✅ Preprocessor cargado")
    
    def preprocess_audio_optimized(self, audio_path: str) -> np.ndarray:
        """Preprocesar audio."""
        import soundfile as sf
        import librosa
        
        audio, sr = sf.read(audio_path)
        
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
        
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        audio = audio / (np.abs(audio).max() + 1e-8)
        audio_batch = audio.reshape(1, -1).astype(np.float32)
        
        return audio_batch
    
    def extract_features_ultra_optimized(self, processed_audio: np.ndarray) -> np.ndarray:
        """Extraer features."""
        audio_shape = np.array([[processed_audio.shape[1]]], dtype=np.int64)
        encoder_inputs = {
            'audio': processed_audio,
            'audio_shape': audio_shape
        }
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        features = encoder_outputs[0]
        
        projection_input_name = self.projection_session.get_inputs()[0].name
        projection_inputs = {projection_input_name: features}
        projection_outputs = self.projection_session.run(None, projection_inputs)
        features = projection_outputs[0]
        
        return features
    
    def generate_caption_optimized(self, audio_features: np.ndarray) -> str:
        """Generar caption."""
        if audio_features.shape[1] == 256:
            encoder_hidden_states = audio_features.transpose(0, 2, 1)
        else:
            encoder_hidden_states = audio_features
        
        bos_token_to_use = self.special_tokens['clotho_bos_id']
        beams = [([bos_token_to_use], 0.0)]
        finished_beams = []
        
        for step in range(self.max_length):
            if not beams:
                break
            
            new_beams = []
            
            for sequence, prev_sum_log_probs in beams:
                if sequence[-1] == self.special_tokens['eos_token_id']:
                    finished_beams.append((sequence, prev_sum_log_probs))
                    continue
                
                input_ids = np.array([sequence], dtype=np.int64)
                decoder_inputs = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': encoder_hidden_states.astype(np.float32)
                }
                
                try:
                    decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                    logits = decoder_outputs[0]
                    last_token_logits = logits[0, -1, :]
                except Exception as e:
                    logger.error(f"Error en decoder: {e}")
                    break
                
                if step + 1 < self.min_pred_size:
                    last_token_logits[self.special_tokens['eos_token_id']] = -float('inf')
                
                if step >= self.max_length - 1:
                    last_token_logits[:] = -float('inf')
                    last_token_logits[self.special_tokens['eos_token_id']] = 0.0
                
                max_logit = np.max(last_token_logits)
                shifted_logits = last_token_logits - max_logit
                log_sum_exp = np.log(np.sum(np.exp(shifted_logits)))
                log_probs = shifted_logits - log_sum_exp
                
                beam_size = 2 if self.jetson_nano_mode else 3
                top_indices = np.argpartition(log_probs, -beam_size)[-beam_size:]
                top_indices = top_indices[np.argsort(log_probs[top_indices])[::-1]]
                
                for token_id in top_indices:
                    new_sequence = sequence + [int(token_id)]
                    new_sum_log_probs = prev_sum_log_probs + log_probs[token_id]
                    new_beams.append((new_sequence, new_sum_log_probs))
            
            new_beams_with_avg = []
            for seq, sum_score in new_beams:
                length = len(seq) - 1 if seq[0] == bos_token_to_use else len(seq)
                avg_score = sum_score / max(1, length)
                new_beams_with_avg.append((seq, sum_score, avg_score))
            
            new_beams_with_avg.sort(key=lambda x: x[2], reverse=True)
            beam_size = 2 if self.jetson_nano_mode else 3
            beams = [(seq, sum_score) for seq, sum_score, _ in new_beams_with_avg[:beam_size]]
        
        finished_beams.extend(beams)
        
        if finished_beams:
            best_beam = None
            best_avg_score = float('-inf')
            
            for seq, sum_score in finished_beams:
                length = len(seq) - 1 if seq[0] == bos_token_to_use else len(seq)
                avg_score = sum_score / max(1, length)
                
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_beam = (seq, sum_score)
            
            generated_tokens = best_beam[0]
        else:
            generated_tokens = [bos_token_to_use]
        
        tokens_to_decode = generated_tokens[1:] if generated_tokens[0] == bos_token_to_use else generated_tokens
        
        try:
            if hasattr(self.tokenizer, 'decode'):
                caption = self.tokenizer.decode(tokens_to_decode)
            else:
                words = []
                for token_id in tokens_to_decode:
                    try:
                        if hasattr(self.tokenizer, 'id_to_token'):
                            word = self.tokenizer.id_to_token(token_id)
                        elif hasattr(self.tokenizer, 'convert_ids_to_tokens'):
                            word = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                        else:
                            continue
                        
                        if word and word not in ['<pad>', '<bos>', '<eos>', '<unk>', '</s>']:
                            words.append(word)
                    except:
                        pass
                caption = " ".join(words)
            
            caption = caption.replace('</s>', '').replace('<pad>', '').strip()
            return caption
            
        except Exception as e:
            logger.error(f"Error decodificando: {e}")
            return "[ERROR]"
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Pipeline completo."""
        start_time = time.time()
        
        try:
            processed_audio = self.preprocess_audio_optimized(audio_path)
            audio_features = self.extract_features_ultra_optimized(processed_audio)
            caption = self.generate_caption_optimized(audio_features)
            
            total_time = time.time() - start_time
            
            return {
                'audio_path': audio_path,
                'caption': caption,
                'total_time': total_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'audio_path': audio_path,
                'error': str(e),
                'success': False
            }

def main():
    """Función principal."""
    logger.info("🎯 DEMO: T5 ONNX COMPLETAMENTE OPTIMIZADO")
    logger.info("=" * 60)
    
    # Crear sistema
    system = T5ONNXFullyOptimized()
    
    # Buscar archivos de audio
    test_file_candidates = [
        "data/voice.wav",
        "data/sample.wav", 
        "data/Bicycle_Bell.wav"
    ]
    
    test_files = []
    base_dir = Path(__file__).parent.parent.parent
    for candidate in test_file_candidates:
        full_path = str(base_dir / candidate)
        if os.path.exists(full_path):
            test_files.append(full_path)
            logger.info(f"✅ Archivo encontrado: {candidate}")
    
    if not test_files:
        logger.warning("⚠️ No se encontraron archivos de audio de prueba")
        return
    
    # Ejecutar predicciones
    results = []
    for audio_file in test_files[:3]:
        logger.info(f"\n🎵 Procesando: {os.path.basename(audio_file)}")
        result = system.predict(audio_file)
        results.append(result)
        
        if result['success']:
            logger.info(f"✅ Caption: '{result['caption']}'")
            logger.info(f"⏱️  Tiempo: {result['total_time']:.3f}s")
        else:
            logger.error(f"❌ Error: {result['error']}")
    
    # Resumen
    logger.info("\n📋 RESUMEN:")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['success']]
    if successful:
        for result in successful:
            filename = os.path.basename(result['audio_path'])
            logger.info(f"✅ {filename}: '{result['caption']}' ({result['total_time']:.3f}s)")
        
        avg_time = sum(r['total_time'] for r in successful) / len(successful)
        logger.info(f"\n⚡ Tiempo promedio: {avg_time:.3f}s")

if __name__ == "__main__":
    main()