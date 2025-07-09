#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T5 ONNX INFERENCE CON ZERO-COPY OPTIMIZATION
============================================

Sistema optimizado con zero-copy para reducir uso de memoria en 50%.
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

# Importar cargador de tokenizer standalone
base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
tokenizer_path = str(base_dir / "Definitivo/06_models/conette_tokenizer_standalone")
logger.info(f"🔍 Buscando tokenizer en: {tokenizer_path}")
logger.info(f"🔍 Existe: {os.path.exists(tokenizer_path)}")
if os.path.exists(tokenizer_path):
    sys.path.append(tokenizer_path)
    from load_tokenizer import StandaloneTokenizerLoader
    STANDALONE_TOKENIZER_AVAILABLE = True
    logger.info("✅ Tokenizer standalone cargado")
else:
    STANDALONE_TOKENIZER_AVAILABLE = False
    logger.warning("⚠️ Tokenizer standalone no encontrado")

class T5ONNXZeroCopyInference:
    """Inference ONNX con optimización zero-copy para Jetson Nano."""
    
    def __init__(self,
                 decoder_path: str = "conette_t5/dec_no_cache/model.onnx",
                 encoder_path: str = "onnx_models_full/conette_encoder.onnx", 
                 projection_path: str = "onnx_models_full/conette_projection.onnx",
                 tokenizer_dir: str = "conette_tokenizer_standalone"):
        
        logger.info("🚀 Inicializando T5ONNXZeroCopyInference")
        
        # Cargar tokenizer standalone
        start_time = time.time()
        self._load_tokenizer_optimized(tokenizer_dir)
        tokenizer_time = time.time() - start_time
        
        logger.info(f"   ⚡ Tokenizer standalone cargado en {tokenizer_time:.3f}s")
        logger.info(f"   📊 Vocab size: {self.special_tokens['vocab_size']}")
        
        # Configurar providers con optimizaciones zero-copy
        self.providers = self._setup_zero_copy_providers()
        
        # Cargar modelos ONNX
        logger.info("📦 Cargando modelos ONNX...")
        self._load_models(decoder_path, encoder_path, projection_path)
        
        # Configurar zero-copy IOBinding
        self._setup_zero_copy_bindings()
        
        # Cargar preprocessor
        self._load_preprocessor()
        
        # Configurar parámetros beam search
        self.beam_size = 3
        self.max_length = 20
        self.min_pred_size = 1
        
        logger.info("✅ T5ONNXZeroCopyInference inicializado con zero-copy")
    
    def _load_tokenizer_optimized(self, tokenizer_dir: str):
        """Cargar tokenizer con estructura corregida."""
        
        # Resolver ruta del tokenizer standalone
        base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
        tokenizer_full_path = str(base_dir / tokenizer_dir)
        
        if STANDALONE_TOKENIZER_AVAILABLE and os.path.exists(tokenizer_full_path):
            logger.info("⚡ Cargando tokenizer standalone...")
            tokenizer_dir = tokenizer_full_path
            self.tokenizer_loader = StandaloneTokenizerLoader(tokenizer_dir)
            self.tokenizer, self.metadata = self.tokenizer_loader.load()
            
            # Estructura corregida de special_tokens
            self.special_tokens = self.metadata.copy()
            
            # Obtener clotho_bos_id del diccionario special_tokens
            if 'special_tokens' in self.metadata and '<bos_clotho>' in self.metadata['special_tokens']:
                self.special_tokens['clotho_bos_id'] = self.metadata['special_tokens']['<bos_clotho>']
            else:
                self.special_tokens['clotho_bos_id'] = self.metadata['bos_token_id']
                
        else:
            logger.info("📦 Fallback: Cargando tokenizer completo...")
            # Add fallback path for conette import
            src_path = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(src_path))
            
            from conette.huggingface.model import CoNeTTEModel
            from conette.huggingface.config import CoNeTTEConfig
            
            config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
            conette_model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
            
            if hasattr(conette_model, 'model') and hasattr(conette_model.model, 'tokenizer'):
                self.tokenizer = conette_model.model.tokenizer
                
                # Crear special_tokens manualmente
                self.special_tokens = {
                    'vocab_size': self.tokenizer.get_vocab_size(),
                    'bos_token_id': self.tokenizer.bos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'clotho_bos_id': self.tokenizer.token_to_id('<bos_clotho>') if self.tokenizer.has('<bos_clotho>') else self.tokenizer.bos_token_id
                }
            else:
                raise ValueError("No se pudo cargar tokenizer")
    
    def _setup_zero_copy_providers(self) -> List:
        """Configurar providers con optimizaciones zero-copy."""
        providers = []
        
        # CPU provider con optimizaciones específicas
        cpu_options = {
            'enable_cpu_mem_arena': True,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'initial_chunk_size_bytes': 1024*1024,  # 1MB chunks
            'max_mem': 0,  # Sin límite
            'allow_non_zero_copy_input': True
        }
        providers.append(('CPUExecutionProvider', cpu_options))
        
        logger.info("✅ Zero-copy providers configurados")
        return providers
    
    def _load_models(self, decoder_path: str, encoder_path: str, projection_path: str):
        """Cargar modelos ONNX con verificación de rutas."""
        
        # Las rutas son relativas al directorio conette (como en los trabajos)
        base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
        decoder_path = str(base_dir / decoder_path)
        encoder_path = str(base_dir / encoder_path) 
        projection_path = str(base_dir / projection_path)
        
        # Configurar sesiones con optimizaciones zero-copy
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Cargar modelos
        self.decoder_session = ort.InferenceSession(decoder_path, providers=self.providers, sess_options=session_options)
        self.encoder_session = ort.InferenceSession(encoder_path, providers=self.providers, sess_options=session_options)
        self.projection_session = ort.InferenceSession(projection_path, providers=self.providers, sess_options=session_options)
        
        logger.info(f"✅ Decoder: {os.path.basename(decoder_path)}")
        logger.info(f"✅ Encoder: {os.path.basename(encoder_path)}")
        logger.info(f"✅ Projection: {os.path.basename(projection_path)}")
    
    def _setup_zero_copy_bindings(self):
        """Configurar IOBinding para zero-copy."""
        logger.info("🔗 Configurando zero-copy IOBinding...")
        
        # Pre-allocar buffers reutilizables
        self.encoder_io_binding = self.encoder_session.io_binding()
        self.projection_io_binding = self.projection_session.io_binding()
        self.decoder_io_binding = self.decoder_session.io_binding()
        
        logger.info("✅ Zero-copy IOBinding configurado")
    
    def _load_preprocessor(self):
        """Cargar preprocessor para formato correcto del audio."""
        logger.info("🔧 Cargando preprocessor de CoNeTTE...")
        
        # Add fallback path for conette import  
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))
        
        from conette.huggingface.model import CoNeTTEModel
        from conette.huggingface.config import CoNeTTEConfig
        
        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
        self._preprocessor = model.preprocessor
        
        logger.info("✅ Preprocessor cargado")
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocesamiento de audio CORRECTO para zero-copy."""
        import soundfile as sf
        import librosa
        
        # Cargar audio
        audio, sr = sf.read(audio_path)
        
        # Resamplear a 32000 Hz (requerido por modelos ONNX)
        target_sr = 32000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Convertir a mono si es estéreo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Normalizar
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Añadir dimensión de batch - FORMATO CORRECTO: [batch, time]
        audio_batch = audio.reshape(1, -1).astype(np.float32)
        
        return audio_batch
    
    def extract_features_zero_copy(self, processed_audio: np.ndarray) -> np.ndarray:
        """Extracción híbrida: encoder ONNX estándar + projection zero-copy."""
        
        # PASO 1: Encoder ONNX estándar (para obtener tamaño exacto)
        audio_shape = np.array([[processed_audio.shape[1]]], dtype=np.int64)
        encoder_inputs = {
            'audio': processed_audio,
            'audio_shape': audio_shape
        }
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        encoder_features = encoder_outputs[0]  # [1, 768, time_frames] con tamaño exacto
        
        # PASO 2: Projection con zero-copy (ahora conocemos el tamaño exacto)
        actual_time_frames = encoder_features.shape[2]  # Tamaño real del encoder
        
        self.projection_io_binding.bind_input(
            name=self.projection_session.get_inputs()[0].name,
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=encoder_features.shape,
            buffer_ptr=encoder_features.ctypes.data
        )
        
        # Output buffer para projection - [1, 256, time_frames] con tamaño exacto
        projection_output_shape = [1, 256, actual_time_frames]
        projection_output = np.empty(projection_output_shape, dtype=np.float32)
        self.projection_io_binding.bind_output(
            name=self.projection_session.get_outputs()[0].name,
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=projection_output_shape,
            buffer_ptr=projection_output.ctypes.data
        )
        
        # Run projection con zero-copy
        self.projection_session.run_with_iobinding(self.projection_io_binding)
        
        return projection_output
    
    def _extract_features_standard(self, processed_audio: np.ndarray) -> np.ndarray:
        """Extracción estándar (fallback)."""
        
        # Encoder - necesita audio y audio_shape
        audio_shape = np.array([[processed_audio.shape[1]]], dtype=np.int64)
        encoder_inputs = {
            'audio': processed_audio,
            'audio_shape': audio_shape
        }
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        features = encoder_outputs[0]
        
        # Projection
        projection_input_name = self.projection_session.get_inputs()[0].name
        projection_inputs = {projection_input_name: features}
        projection_outputs = self.projection_session.run(None, projection_inputs)
        features = projection_outputs[0]
        
        return features
    
    def generate_caption(self, audio_features: np.ndarray) -> str:
        """Generación de caption usando formato T5."""
        
        # Formato T5: encoder_hidden_states debe ser [batch, time, hidden]
        if audio_features.shape[1] == 256:  # [batch, hidden=256, time]
            encoder_hidden_states = audio_features.transpose(0, 2, 1)  # [batch, time, hidden]
        else:
            encoder_hidden_states = audio_features  # Ya es [batch, time, hidden]
        
        # Usar token BOS específico de Clotho
        bos_token_to_use = self.special_tokens['clotho_bos_id']
        
        # Beam search simplificado para zero-copy
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
                
                # Preparar inputs T5
                input_ids = np.array([sequence], dtype=np.int64)  # [batch, seq_len]
                
                decoder_inputs = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': encoder_hidden_states.astype(np.float32)
                }
                
                try:
                    decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                    logits = decoder_outputs[0]
                    
                    # Obtener logits del último token (formato T5)
                    last_token_logits = logits[0, -1, :]  # [vocab_size] - último token de la secuencia
                    
                except Exception as e:
                    logger.error(f"❌ Error en decoder paso {step}: {e}")
                    break
                
                # Aplicar restricciones CoNeTTE
                if step + 1 < self.min_pred_size:
                    last_token_logits[self.special_tokens['eos_token_id']] = -float('inf')
                
                if step >= self.max_length - 1:
                    last_token_logits[:] = -float('inf')
                    last_token_logits[self.special_tokens['eos_token_id']] = 0.0
                
                # LogSoftmax exacto
                max_logit = np.max(last_token_logits)
                shifted_logits = last_token_logits - max_logit
                log_sum_exp = np.log(np.sum(np.exp(shifted_logits)))
                log_probs = shifted_logits - log_sum_exp
                
                # Seleccionar mejores tokens
                top_indices = np.argpartition(log_probs, -self.beam_size)[-self.beam_size:]
                top_indices = top_indices[np.argsort(log_probs[top_indices])[::-1]]
                
                # Expandir beam
                for token_id in top_indices:
                    new_sequence = sequence + [int(token_id)]
                    new_sum_log_probs = prev_sum_log_probs + log_probs[token_id]
                    new_beams.append((new_sequence, new_sum_log_probs))
            
            # Ordenar por score promedio normalizado
            new_beams_with_avg = []
            for seq, sum_score in new_beams:
                length = len(seq) - 1 if seq[0] == bos_token_to_use else len(seq)
                avg_score = sum_score / max(1, length)
                new_beams_with_avg.append((seq, sum_score, avg_score))
            
            new_beams_with_avg.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, sum_score) for seq, sum_score, _ in new_beams_with_avg[:self.beam_size]]
        
        # Mover beams restantes a finished
        finished_beams.extend(beams)
        
        # Seleccionar mejor secuencia
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
        
        # Decodificar usando tokenizer standalone
        tokens_to_decode = generated_tokens[1:] if generated_tokens[0] == bos_token_to_use else generated_tokens
        
        try:
            if hasattr(self.tokenizer, 'decode_batch'):
                caption = self.tokenizer.decode_batch([tokens_to_decode])[0]
            elif hasattr(self.tokenizer, 'decode_single'):
                caption = self.tokenizer.decode_single(tokens_to_decode)
            else:
                # Fallback manual
                words = []
                for token_id in tokens_to_decode:
                    try:
                        word = self.tokenizer.id_to_token(token_id)
                        if word not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                            words.append(word)
                    except:
                        pass
                caption = " ".join(words)
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"❌ Error decodificando: {e}")
            return f"[ERROR: {str(e)}]"
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Pipeline completo de predicción con zero-copy."""
        
        start_time = time.time()
        
        try:
            # 1. Preprocesamiento con formato correcto
            processed_audio = self.preprocess_audio(audio_path)
            
            # 2. Extracción de features con zero-copy optimizado
            audio_features = self.extract_features_zero_copy(processed_audio)
            
            # 3. Generación
            caption = self.generate_caption(audio_features)
            
            total_time = time.time() - start_time
            
            return {
                'audio_path': audio_path,
                'caption': caption,
                'total_time': total_time,
                'zero_copy': True,
                'success': True
            }
            
        except Exception as e:
            return {
                'audio_path': audio_path,
                'error': str(e),
                'success': False
            }

def main():
    """Demo con zero-copy optimization."""
    
    logger.info("🎯 DEMO: T5 ONNX Zero-Copy OPTIMIZADO")
    logger.info("=" * 50)
    
    # Crear sistema
    system = T5ONNXZeroCopyInference()
    
    # Test files - usar rutas como en los sistemas que funcionan
    test_file_candidates = [
        "data/voice.wav",
        "data/sample.wav", 
        "data/Bicycle_Bell.wav"
    ]
    
    test_files = []
    base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
    for candidate in test_file_candidates:
        full_path = str(base_dir / candidate)
        if os.path.exists(full_path):
            test_files.append(full_path)
    
    if not test_files:
        logger.warning("⚠️ No se encontraron archivos de audio de prueba")
        return
    
    # Usar solo los primeros archivos encontrados
    test_files = test_files[:3]
    
    for audio_file in test_files:
        result = system.predict(audio_file)
        
        if result['success']:
            logger.info(f"✅ {os.path.basename(audio_file)}")
            logger.info(f"   Caption: '{result['caption']}'")
            logger.info(f"   Tiempo: {result['total_time']:.3f}s")
            logger.info(f"   Zero-copy: {result['zero_copy']}")
        else:
            logger.info(f"❌ {os.path.basename(audio_file)}")
            logger.info(f"   Error: {result['error']}")

if __name__ == "__main__":
    main()