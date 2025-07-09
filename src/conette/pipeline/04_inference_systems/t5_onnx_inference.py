#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA T5 ONNX CORREGIDO
========================

Sistema de inferencia corregido que usa los inputs exactos que esperan los modelos ONNX exportados.
Basado en el debug de los modelos reales.
"""

import os
import sys
import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add current source to path to use local conette instead of installed package
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent  # Go up to src/conette level
sys.path.insert(0, str(src_dir.parent))  # Add src/ to path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar cargador de tokenizer standalone
tokenizer_path = "06_models/conette_tokenizer_standalone"
if os.path.exists(tokenizer_path):
    sys.path.append(tokenizer_path)
    from load_tokenizer import StandaloneTokenizerLoader
    STANDALONE_TOKENIZER_AVAILABLE = True
else:
    STANDALONE_TOKENIZER_AVAILABLE = False
    logger.warning("⚠️ Tokenizer standalone no encontrado")

class T5ONNXCorrected:
    """Sistema T5 ONNX que usa los inputs/outputs correctos de los modelos exportados."""
    
    def __init__(self,
                 encoder_path: str = "onnx_models_full/conette_encoder.onnx",
                 projection_path: str = "onnx_models_full/conette_projection.onnx",
                 decoder_path: str = "conette_t5/dec_no_cache/model.onnx",
                 tokenizer_dir: str = "conette_tokenizer_standalone"):
        
        logger.info("🚀 Inicializando T5ONNXCorrected...")
        
        # 1. Cargar tokenizer standalone
        self._load_tokenizer_optimized(tokenizer_dir)
        
        # 2. Configurar providers
        self.providers = ['CPUExecutionProvider']
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            self.providers.insert(0, 'CUDAExecutionProvider')
            logger.info("✅ CUDA disponible")
        
        # 3. Cargar modelos ONNX
        self._load_models(encoder_path, projection_path, decoder_path)
        
        # 4. Configurar parámetros beam search
        self.beam_size = 3
        self.max_length = 20
        self.min_pred_size = 1
        
        logger.info("✅ T5ONNXCorrected inicializado")
    
    def _load_tokenizer_optimized(self, tokenizer_dir: str):
        """Cargar tokenizer con máxima optimización."""
        
        # Resolver ruta del tokenizer 
        base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
        tokenizer_full_path = str(base_dir / tokenizer_dir)
        
        if STANDALONE_TOKENIZER_AVAILABLE and os.path.exists(tokenizer_full_path):
            logger.info("⚡ Cargando tokenizer standalone...")
            self.tokenizer_loader = StandaloneTokenizerLoader(tokenizer_full_path)
            self.tokenizer, self.metadata = self.tokenizer_loader.load()
            self.special_tokens = self.tokenizer_loader.get_special_tokens()
        else:
            logger.info("📦 Fallback: Cargando tokenizer completo...")
            from conette.huggingface.model import CoNeTTEModel
            from conette.huggingface.config import CoNeTTEConfig
            
            config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
            conette_model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
            
            if hasattr(conette_model, 'model') and hasattr(conette_model.model, 'tokenizer'):
                self.tokenizer = conette_model.model.tokenizer
                
                # Crear special_tokens manualmente
                self.special_tokens = {
                    'bos_id': self.tokenizer.bos_token_id,
                    'eos_id': self.tokenizer.eos_token_id,
                    'pad_id': self.tokenizer.pad_token_id,
                    'vocab_size': self.tokenizer.get_vocab_size(),
                    'clotho_bos_id': self.tokenizer.token_to_id('<bos_clotho>') if self.tokenizer.has('<bos_clotho>') else self.tokenizer.bos_token_id
                }
            else:
                raise ValueError("No se pudo cargar tokenizer")
        
        logger.info(f"   📊 Vocab size: {self.special_tokens['vocab_size']}")
    
    def _load_models(self, encoder_path: str, projection_path: str, decoder_path: str):
        """Cargar modelos ONNX con verificación de rutas."""
        
        logger.info("📦 Cargando modelos ONNX...")
        
        # Resolver rutas relativas desde directorio conette
        base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
        encoder_path = str(base_dir / encoder_path)
        projection_path = str(base_dir / projection_path)
        decoder_path = str(base_dir / decoder_path)
        
        # Cargar encoder
        if os.path.exists(encoder_path):
            self.encoder_session = ort.InferenceSession(encoder_path, providers=self.providers)
            logger.info(f"✅ Encoder: {encoder_path}")
        else:
            logger.error(f"❌ Encoder no encontrado: {encoder_path}")
            self.encoder_session = None
        
        # Cargar projection
        if os.path.exists(projection_path):
            self.projection_session = ort.InferenceSession(projection_path, providers=self.providers)
            logger.info(f"✅ Projection: {projection_path}")
        else:
            logger.error(f"❌ Projection no encontrado: {projection_path}")
            self.projection_session = None
        
        # Cargar decoder
        decoder_candidates = [decoder_path]
        
        self.decoder_session = None
        for path in decoder_candidates:
            if os.path.exists(path):
                try:
                    self.decoder_session = ort.InferenceSession(path, providers=self.providers)
                    logger.info(f"✅ Decoder: {path}")
                    self.decoder_path = path
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando {path}: {e}")
        
        if not self.decoder_session:
            logger.error("❌ No se encontró ningún decoder válido")
            raise RuntimeError("Decoder no disponible")
        
        # Verificar componentes críticos
        if not self.encoder_session or not self.projection_session:
            missing = []
            if not self.encoder_session: missing.append("encoder")
            if not self.projection_session: missing.append("projection")
            logger.error(f"❌ Componentes faltantes: {missing}")
            raise RuntimeError(f"Modelos faltantes: {missing}")
    
    def preprocess_audio_corrected(self, audio_path: str) -> np.ndarray:
        """Preprocesamiento de audio CORRECTO - igual al sistema que funciona."""
        
        logger.info(f"🎵 Preprocesando audio: {audio_path}")
        
        try:
            import soundfile as sf
            import librosa
            
            # Cargar audio
            audio, sr = sf.read(audio_path)
            
            # Resamplear a 32000 Hz (requerido por modelos ONNX)
            target_sr = 32000
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                logger.info(f"   🔄 Remuestreado de {sr}Hz a {target_sr}Hz")
            
            # Convertir a mono si es estéreo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Normalizar
            audio = audio / (np.abs(audio).max() + 1e-8)
            
            # Añadir dimensión de batch - FORMATO CORRECTO: [batch, time]
            audio_batch = audio.reshape(1, -1).astype(np.float32)
            
            logger.info(f"   📊 Audio preprocesado: {audio_batch.shape}")
            return audio_batch
            
        except Exception as e:
            logger.error(f"❌ Error preprocesando audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def extract_features_corrected(self, processed_audio: np.ndarray) -> np.ndarray:
        """Extracción de features usando los inputs correctos de los modelos ONNX."""
        
        logger.info("🔧 Extrayendo features...")
        
        try:
            # 1. Encoder ONNX - Necesita 'audio' y 'audio_shape'
            audio_shape = np.array([[processed_audio.shape[1]]], dtype=np.int64)  # [batch_size, 1]
            encoder_inputs = {
                'audio': processed_audio,
                'audio_shape': audio_shape
            }
            encoder_outputs = self.encoder_session.run(None, encoder_inputs)
            features = encoder_outputs[0]  # frame_embs
            
            logger.info(f"   📊 Encoder output: {features.shape}")
            
            # 2. Projection ONNX - Necesita 'encoder_features'
            projection_inputs = {'encoder_features': features}
            projection_outputs = self.projection_session.run(None, projection_inputs)
            projected_features = projection_outputs[0]  # projected_features
            
            logger.info(f"   📊 Projection output: {projected_features.shape}")
            
            return projected_features
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_caption_corrected(self, audio_features: np.ndarray) -> str:
        """Generación usando el formato correcto según el decoder disponible."""
        
        logger.info("🔍 Iniciando generación de caption...")
        
        try:
            # Detectar qué tipo de decoder tenemos
            decoder_inputs = [inp.name for inp in self.decoder_session.get_inputs()]
            
            if 'input_ids' in decoder_inputs and 'encoder_hidden_states' in decoder_inputs:
                # Decoder corrected con formato T5
                return self._generate_with_t5_decoder(audio_features)
            elif 'frame_embs' in decoder_inputs and 'caps_in' in decoder_inputs:
                # Decoder original CoNeTTE
                return self._generate_with_conette_decoder(audio_features)
            else:
                logger.error(f"❌ Formato de decoder desconocido: {decoder_inputs}")
                return "[ERROR: Formato de decoder desconocido]"
                
        except Exception as e:
            logger.error(f"❌ Error generando caption: {e}")
            return f"[ERROR: {str(e)}]"
    
    def _generate_with_t5_decoder(self, audio_features: np.ndarray) -> str:
        """Generación con decoder formato T5."""
        
        logger.info("   🎯 Usando decoder T5...")
        
        # Ajustar formato para T5: [batch, time, hidden] 
        if audio_features.shape[1] == 256:  # [batch, hidden=256, time]
            encoder_hidden_states = audio_features.transpose(0, 2, 1)  # [batch, time, hidden]
        else:
            encoder_hidden_states = audio_features  # Ya es [batch, time, hidden]
        
        # Token BOS - manejar diferentes formatos de special_tokens
        if 'clotho_bos_id' in self.special_tokens:
            bos_token = self.special_tokens['clotho_bos_id']
        elif 'special_tokens' in self.special_tokens and '<bos_clotho>' in self.special_tokens['special_tokens']:
            bos_token = self.special_tokens['special_tokens']['<bos_clotho>']
        else:
            bos_token = self.special_tokens.get('bos_token_id', self.special_tokens.get('bos_id', 1))
        
        # Beam search simple
        beams = [([bos_token], 0.0)]
        finished_beams = []
        
        for step in range(self.max_length):
            if not beams:
                break
            
            new_beams = []
            
            for sequence, prev_score in beams:
                eos_token = self.special_tokens.get('eos_token_id', self.special_tokens.get('eos_id', 2))
                if sequence[-1] == eos_token:
                    finished_beams.append((sequence, prev_score))
                    continue
                
                # Preparar inputs T5
                input_ids = np.array([sequence], dtype=np.int64)
                decoder_inputs = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': encoder_hidden_states.astype(np.float32)
                }
                
                # Ejecutar decoder
                decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                logits = decoder_outputs[0]
                
                # Obtener logits del último token
                last_token_logits = logits[0, -1, :]
                
                # Restricciones
                eos_token = self.special_tokens.get('eos_token_id', self.special_tokens.get('eos_id', 2))
                if step + 1 < self.min_pred_size:
                    last_token_logits[eos_token] = -float('inf')
                
                # LogSoftmax
                max_logit = np.max(last_token_logits)
                log_probs = last_token_logits - max_logit - np.log(np.sum(np.exp(last_token_logits - max_logit)))
                
                # Top-k tokens
                top_indices = np.argpartition(log_probs, -self.beam_size)[-self.beam_size:]
                
                for token_id in top_indices:
                    new_sequence = sequence + [int(token_id)]
                    new_score = prev_score + log_probs[token_id]
                    new_beams.append((new_sequence, new_score))
            
            # Mantener mejores beams
            new_beams.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = new_beams[:self.beam_size]
        
        # Seleccionar mejor
        finished_beams.extend(beams)
        if finished_beams:
            best_sequence = max(finished_beams, key=lambda x: x[1] / len(x[0]))[0]
        else:
            best_sequence = [bos_token]
        
        # Decodificar
        return self._decode_tokens(best_sequence, bos_token)
    
    def _generate_with_conette_decoder(self, audio_features: np.ndarray) -> str:
        """Generación con decoder formato CoNeTTE original."""
        
        logger.info("   🎯 Usando decoder CoNeTTE...")
        
        # Ajustar formato para CoNeTTE: [time, batch, hidden]
        if audio_features.shape[1] == 256:  # [batch, hidden=256, time]
            frame_embs = audio_features.transpose(2, 0, 1)  # [time, batch, hidden]
        else:
            frame_embs = audio_features.transpose(1, 0, 2)  # [time, batch, hidden]
        
        # Token BOS
        if 'special_tokens' in self.special_tokens and '<bos_clotho>' in self.special_tokens['special_tokens']:
            bos_token = self.special_tokens['special_tokens']['<bos_clotho>']
        else:
            bos_token = self.special_tokens.get('bos_token_id', 1)
        
        # Generar secuencia simple (sin beam search complejo por ahora)
        sequence = [bos_token]
        
        for step in range(self.max_length):
            # Preparar inputs CoNeTTE
            caps_in = np.array([sequence], dtype=np.int64).T  # [seq_len, batch]
            
            decoder_inputs = {
                'frame_embs': frame_embs.astype(np.float32),
                'frame_embs_pad_mask': None,  # TODO: crear mask apropiado
                'caps_in': caps_in
            }
            
            # Filtrar inputs None
            decoder_inputs = {k: v for k, v in decoder_inputs.items() if v is not None}
            
            try:
                # Ejecutar decoder
                decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                logits = decoder_outputs[0]
                
                # Obtener logits del último token
                last_token_logits = logits[-1, 0, :]  # [vocab_size]
                
                # Seleccionar token más probable
                next_token = np.argmax(last_token_logits)
                sequence.append(int(next_token))
                
                # Parar si es EOS
                eos_token = self.special_tokens.get('eos_token_id', self.special_tokens.get('eos_id', 2))
                if next_token == eos_token:
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error en paso {step}: {e}")
                break
        
        # Decodificar
        return self._decode_tokens(sequence, bos_token)
    
    def _decode_tokens(self, tokens: List[int], bos_token: int) -> str:
        """Decodificar tokens."""
        
        try:
            # Remover BOS/EOS
            tokens_to_decode = tokens[1:] if tokens[0] == bos_token else tokens
            eos_token = self.special_tokens.get('eos_token_id', self.special_tokens.get('eos_id', 2))
            if tokens_to_decode and tokens_to_decode[-1] == eos_token:
                tokens_to_decode = tokens_to_decode[:-1]
            
            # Decodificar
            if hasattr(self.tokenizer, 'decode_batch'):
                return self.tokenizer.decode_batch([tokens_to_decode])[0].strip()
            elif hasattr(self.tokenizer, 'decode_single'):
                return self.tokenizer.decode_single(tokens_to_decode).strip()
            else:
                # Fallback manual
                words = []
                for token_id in tokens_to_decode:
                    try:
                        word = self.tokenizer.id_to_token(token_id)
                        if word and word not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                            words.append(word)
                    except:
                        pass
                return " ".join(words)
            
        except Exception as e:
            logger.error(f"❌ Error decodificando: {e}")
            return f"[DECODE_ERROR: {str(e)}]"
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Pipeline completo de predicción."""
        
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Prediciendo para: {audio_path}")
            
            # 1. Preprocesar audio
            processed_audio = self.preprocess_audio_corrected(audio_path)
            
            # 2. Extraer features
            audio_features = self.extract_features_corrected(processed_audio)
            
            # 3. Generar caption
            caption = self.generate_caption_corrected(audio_features)
            
            total_time = time.time() - start_time
            
            result = {
                'audio_path': audio_path,
                'caption': caption,
                'total_time': total_time,
                'success': True
            }
            
            logger.info(f"✅ '{caption}' ({total_time:.3f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {
                'audio_path': audio_path,
                'error': str(e),
                'success': False
            }

def main():
    """Test del sistema corregido."""
    
    logger.info("🧪 TEST: SISTEMA T5 ONNX CORREGIDO")
    logger.info("=" * 50)
    
    try:
        # Crear sistema
        system = T5ONNXCorrected()
        
        # Test files - ajustar según tu entorno
        test_files = [
            "/workspace/conette/data/sample.wav", 
            "/workspace/conette/data/voice.wav",
            "/home/gsolsan/conette-audio-captioning/src/conette/data/sample.wav",
            "/home/gsolsan/conette-audio-captioning/src/conette/data/voice.wav"
        ]
        
        # Filtrar solo archivos que existen
        existing_files = [f for f in test_files if Path(f).exists()]
        if not existing_files:
            logger.warning("⚠️ No se encontraron archivos de audio de prueba")
            logger.info("💡 Coloca archivos .wav en /workspace/conette/data/ para probar")
            return True
            
        test_files = existing_files[:2]  # Usar solo los primeros 2 encontrados
        results = []
        
        for audio_file in test_files:
            if Path(audio_file).exists():
                result = system.predict(audio_file)
                results.append(result)
            else:
                logger.warning(f"⚠️ No encontrado: {audio_file}")
        
        # Resumen
        logger.info("\n📋 RESUMEN:")
        successful = [r for r in results if r['success']]
        logger.info(f"✅ Exitosos: {len(successful)}/{len(results)}")
        
        if successful:
            avg_time = sum(r['total_time'] for r in successful) / len(successful)
            logger.info(f"⚡ Tiempo promedio: {avg_time:.3f}s")
            
            for result in successful:
                filename = Path(result['audio_path']).name
                logger.info(f"🎵 {filename}: '{result['caption']}'")
        
        return len(successful) > 0
        
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)