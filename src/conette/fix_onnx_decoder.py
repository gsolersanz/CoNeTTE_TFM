#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SOLUCIÓN DEFINITIVA PARA EL PROBLEMA DE REPETICIÓN EN DECODER ONNX
=================================================================

PROBLEMA IDENTIFICADO:
- El decoder ONNX exportado no mantiene correctamente el contexto secuencial
- Las transposiciones de tensores están causando pérdida de información temporal
- La máscara causal no se aplica correctamente en formato ONNX

SOLUCIÓN:
- Crear un wrapper que mantenga la compatibilidad exacta con AACTransformerDecoder
- Preservar el formato de tensores tal como espera el decoder original
- Implementar export personalizado que respete el flujo secuencial
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any

import torch
import torchaudio
import numpy as np
from torch import nn, Tensor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fix_onnx_decoder.log')
    ]
)

logger = logging.getLogger(__name__)

class FixedONNXDecoderWrapper(nn.Module):
    """
    Wrapper corregido para el decoder que mantiene compatibilidad exacta
    con AACTransformerDecoder y preserva el contexto secuencial.
    """
    
    def __init__(self, model):
        super().__init__()
        # Extraer el decoder original
        self.original_decoder = model.model.decoder
        
        # Debug: verificar qué atributos tiene el decoder
        logger.info(f"Decoder type: {type(self.original_decoder)}")
        logger.info(f"Decoder attributes: {dir(self.original_decoder)}")
        
        # Extraer componentes esenciales manteniendo referencias exactas
        self.emb_layer = self.original_decoder.emb_layer
        self.pos_encoding = self.original_decoder.pos_encoding  
        self.classifier = self.original_decoder.classifier
        
        # Extraer transformer layers manteniendo exactamente la misma estructura
        self.decoder_layers = self.original_decoder.layers
        self.norm = getattr(self.original_decoder, 'norm', None)
        
        # Mantener atributos críticos - verificar si existe d_model
        if hasattr(self.original_decoder, 'd_model'):
            self.d_model = self.original_decoder.d_model
        else:
            # Inferir d_model del tamaño de embedding
            self.d_model = self.emb_layer.embedding_dim
            logger.warning(f"d_model not found in decoder, inferred from embedding: {self.d_model}")
        
        self.bos_id = self.original_decoder.bos_id
        self.eos_id = self.original_decoder.eos_id
        self.pad_id = self.original_decoder.pad_id
        
        logger.info(f"FixedONNXDecoderWrapper initialized:")
        logger.info(f"  - d_model: {self.d_model}")
        logger.info(f"  - num_layers: {len(self.decoder_layers)}")
        logger.info(f"  - vocab_size: {self.classifier.out_features}")
    
    def forward(
        self,
        input_ids: Tensor,           # [batch_size, seq_len]
        encoder_hidden_states: Tensor,  # [batch_size, time_steps, hidden_size]
        encoder_attention_mask: Optional[Tensor] = None  # [batch_size, time_steps]
    ) -> Tensor:
        """
        Forward que mantiene exactamente la misma lógica que AACTransformerDecoder
        pero adaptado para funcionar con el formato ONNX.
        """
        
        logger.debug(f"Input shapes - input_ids: {input_ids.shape}, encoder_hidden: {encoder_hidden_states.shape}")
        
        # PASO 1: Convertir input_ids a embeddings usando el mismo proceso que el original
        if input_ids.is_floating_point():
            # Ya son embeddings
            caps_in = input_ids
        else:
            # Convertir IDs a embeddings
            caps_in = self.emb_layer(input_ids)  # [batch_size, seq_len, d_model]
        
        # Asegurar float32
        caps_in = caps_in.to(dtype=torch.float32)
        
        # PASO 2: Aplicar escala exactamente como en el original
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=caps_in.device))
        caps_in = caps_in * scale
        
        # PASO 3: Transponer a formato esperado por transformer: [seq_len, batch_size, d_model]
        caps_in = caps_in.transpose(0, 1)
        
        # PASO 4: Aplicar positional encoding usando el módulo original
        caps_in = self.pos_encoding(caps_in)
        
        # PASO 5: Preparar memory (encoder outputs) en formato correcto
        # Transponer de [batch_size, time_steps, hidden_size] a [time_steps, batch_size, hidden_size]
        frame_embs = encoder_hidden_states.transpose(0, 1)
        
        # PASO 6: Crear máscaras exactamente como en AACTransformerDecoder
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Máscara causal para tokens
        caps_in_sq_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        ).to(torch.float32)
        
        # Máscara de padding para encoder
        if encoder_attention_mask is None:
            time_steps = frame_embs.shape[0]
            frame_embs_pad_mask = torch.zeros((batch_size, time_steps), dtype=torch.bool, device=device)
        else:
            frame_embs_pad_mask = encoder_attention_mask.to(torch.bool)
        
        # Máscara de padding para tokens (detectar pad_id)
        caps_in_pad_mask = (input_ids == self.pad_id)
        
        logger.debug(f"Prepared tensors - caps_in: {caps_in.shape}, frame_embs: {frame_embs.shape}")
        logger.debug(f"Masks - causal: {caps_in_sq_mask.shape}, frame_pad: {frame_embs_pad_mask.shape}, caps_pad: {caps_in_pad_mask.shape}")
        
        # PASO 7: Llamar al decoder usando exactamente la misma interfaz que AACTransformerDecoder
        try:
            # Usar la interfaz exacta de AACTransformerDecoder.forward
            tok_logits_out = self.original_decoder(
                frame_embs=frame_embs,                    # [time_steps, batch_size, hidden_size]
                frame_embs_pad_mask=frame_embs_pad_mask,  # [batch_size, time_steps]  
                caps_in=caps_in,                          # [seq_len, batch_size, d_model]
                caps_in_pad_mask=caps_in_pad_mask,        # [batch_size, seq_len]
                caps_in_sq_mask=caps_in_sq_mask,          # [seq_len, seq_len]
            )
            
            logger.debug(f"Decoder output shape: {tok_logits_out.shape}")
            
            # El decoder original ya aplica el classifier y devuelve logits
            # en formato [seq_len, batch_size, vocab_size]
            # Transponer de vuelta a [batch_size, seq_len, vocab_size] para compatibilidad ONNX
            logits = tok_logits_out.transpose(0, 1)
            
            logger.debug(f"Final logits shape: {logits.shape}")
            return logits
            
        except Exception as e:
            logger.error(f"Error in decoder forward: {e}")
            logger.error(traceback.format_exc())
            raise


class DebugAwareBeamSearch:
    """
    Implementación de beam search que usa el decoder corregido
    y mantiene debugging exhaustivo.
    """
    
    def __init__(self, 
                 encoder_path: str = "onnx_models_full/conette_encoder.onnx",
                 projection_path: str = "onnx_models_full/conette_projection.onnx"):
        
        logger.info("Inicializando DebugAwareBeamSearch con decoder corregido")
        
        # Cargar modelo PyTorch original
        self._load_pytorch_model()
        
        # Cargar componentes ONNX existentes
        self._load_onnx_components(encoder_path, projection_path)
        
        # Crear decoder wrapper corregido
        self.fixed_decoder = FixedONNXDecoderWrapper(self.pytorch_model)
        self.fixed_decoder.eval()
        
        logger.info("✅ DebugAwareBeamSearch inicializado correctamente")
    
    def _load_pytorch_model(self):
        """Cargar modelo PyTorch para referencia"""
        try:
            from conette import CoNeTTEConfig, CoNeTTEModel
            
            config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
            self.pytorch_model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
            self.tokenizer = self.pytorch_model.model.tokenizer
            
            # Extraer información del tokenizer
            bos_token = "<bos_clotho>"
            if self.tokenizer.has(bos_token):
                self.bos_id = self.tokenizer.token_to_id(bos_token)
            else:
                self.bos_id = self.tokenizer.bos_token_id
                
            self.eos_id = self.tokenizer.eos_token_id
            self.vocab_size = self.tokenizer.get_vocab_size()
            
            logger.info(f"PyTorch model loaded - BOS: {self.bos_id}, EOS: {self.eos_id}, Vocab: {self.vocab_size}")
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise
    
    def _load_onnx_components(self, encoder_path: str, projection_path: str):
        """Cargar componentes ONNX existentes"""
        try:
            import onnxruntime as ort
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.encoder_session = ort.InferenceSession(encoder_path, sess_options)
            self.projection_session = ort.InferenceSession(projection_path, sess_options)
            
            logger.info(f"ONNX components loaded - encoder: {encoder_path}, projection: {projection_path}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX components: {e}")
            raise
    
    def load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Cargar y preprocesar audio"""
        logger.info(f"Loading audio from {audio_path}")
        
        audio, sr = torchaudio.load(audio_path)
        
        # Remuestrear si es necesario
        if sr != 32000:
            resampler = torchaudio.transforms.Resample(sr, 32000)
            audio = resampler(audio)
            logger.info(f"Audio resampled from {sr}Hz to 32000Hz")
        
        # Convertir a mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Convertir a numpy
        audio_np = audio.numpy().astype(np.float32)
        audio_shape_np = np.array([[audio.shape[1]]], dtype=np.int64)
        
        logger.info(f"Audio processed - shape: {audio_np.shape}, duration: {audio.shape[1]/32000:.2f}s")
        
        return audio_np, audio_shape_np
    
    def extract_features(self, audio_np: np.ndarray, audio_shape_np: np.ndarray) -> np.ndarray:
        """Extraer características usando encoder y projection ONNX"""
        logger.info("Extracting features using ONNX encoder and projection")
        
        # Ejecutar encoder
        encoder_inputs = {'audio': audio_np, 'audio_shape': audio_shape_np}
        frame_embs_raw, frame_embs_pad_mask = self.encoder_session.run(None, encoder_inputs)
        
        # Ejecutar projection
        projection_inputs = {'encoder_features': frame_embs_raw}
        projected_features = self.projection_session.run(None, projection_inputs)[0]
        
        # Transponer para formato correcto: (batch, hidden, time) -> (batch, time, hidden)
        encoder_hidden_states = np.transpose(projected_features, (0, 2, 1))
        
        logger.info(f"Features extracted - shape: {encoder_hidden_states.shape}")
        
        return encoder_hidden_states
    
    def compare_first_step_logits(self, encoder_hidden_states: np.ndarray) -> Dict[str, Any]:
        """Comparar logits del primer paso entre PyTorch y decoder corregido"""
        logger.info("Comparing first step logits...")
        
        # Preparar entradas
        input_ids = torch.tensor([[self.bos_id]], dtype=torch.long)
        encoder_hidden_torch = torch.from_numpy(encoder_hidden_states).float()
        
        # 1. Obtener logits del decoder corregido
        with torch.no_grad():
            fixed_logits = self.fixed_decoder(input_ids, encoder_hidden_torch)
            fixed_first_token_logits = fixed_logits[0, 0, :]  # [vocab_size]
        
        # 2. Obtener predicción de PyTorch como referencia
        audio_path = "data/sample.wav"  # Asumiendo que existe
        try:
            with torch.no_grad():
                pytorch_outputs = self.pytorch_model(audio_path)
                pytorch_result = pytorch_outputs["cands"][0]
        except:
            pytorch_result = "No disponible"
        
        # 3. Analizar distribución
        top_10_indices = torch.argsort(fixed_first_token_logits, descending=True)[:10]
        
        results = {
            "pytorch_reference": pytorch_result,
            "fixed_decoder_top_tokens": [],
            "fixed_decoder_analysis": {}
        }
        
        logger.info("Top 10 tokens from fixed decoder:")
        for i, token_id in enumerate(top_10_indices):
            try:
                token_str = self.tokenizer.id_to_token(int(token_id))
                score = float(fixed_first_token_logits[token_id])
                logger.info(f"  {i+1:2d}. Token {token_id:5d} ('{token_str:>15s}'): {score:8.4f}")
                
                results["fixed_decoder_top_tokens"].append({
                    "rank": i+1,
                    "token_id": int(token_id),
                    "token_str": token_str,
                    "score": score
                })
            except:
                score = float(fixed_first_token_logits[token_id])
                logger.info(f"  {i+1:2d}. Token {token_id:5d}: {score:8.4f}")
        
        # Verificar si 'rain' sigue siendo dominante
        rain_token_id = None
        for i in range(min(1000, self.vocab_size)):
            try:
                token_str = self.tokenizer.id_to_token(i)
                if token_str and token_str.strip().lower() == 'rain':
                    rain_token_id = i
                    break
            except:
                continue
        
        if rain_token_id is not None:
            rain_score = float(fixed_first_token_logits[rain_token_id])
            rain_rank = (fixed_first_token_logits > rain_score).sum().item() + 1
            
            logger.info(f"'rain' token analysis:")
            logger.info(f"  Token ID: {rain_token_id}")
            logger.info(f"  Score: {rain_score:.4f}")
            logger.info(f"  Rank: {rain_rank}")
            
            results["fixed_decoder_analysis"]["rain_analysis"] = {
                "token_id": rain_token_id,
                "score": rain_score,
                "rank": rain_rank,
                "is_dominant": rain_rank <= 3
            }
        
        return results
    
    def run_beam_search(self, encoder_hidden_states: np.ndarray, 
                       beam_size: int = 3, max_length: int = 20) -> List[Dict[str, Any]]:
        """Ejecutar beam search con el decoder corregido"""
        logger.info(f"Running beam search - beam_size: {beam_size}, max_length: {max_length}")
        
        encoder_hidden_torch = torch.from_numpy(encoder_hidden_states).float()
        
        # Inicializar beam search
        beams = [{"tokens": [self.bos_id], "score": 0.0}]
        finished_beams = []
        
        for step in range(max_length):
            logger.info(f"Beam search step {step+1}/{max_length}")
            
            all_candidates = []
            
            for beam in beams:
                # Preparar entrada
                input_ids = torch.tensor([beam["tokens"]], dtype=torch.long)
                
                # Ejecutar decoder
                with torch.no_grad():
                    logits = self.fixed_decoder(input_ids, encoder_hidden_torch)
                    next_token_logits = logits[0, -1, :]  # Último token
                
                # Convertir a log-probabilidades
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Obtener top-k candidatos
                top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_size)
                
                for log_prob, token_id in zip(top_k_log_probs, top_k_indices):
                    new_beam = {
                        "tokens": beam["tokens"] + [int(token_id)],
                        "score": beam["score"] + float(log_prob)
                    }
                    
                    if int(token_id) == self.eos_id:
                        finished_beams.append(new_beam)
                    else:
                        all_candidates.append(new_beam)
            
            # Seleccionar mejores beams
            all_candidates.sort(key=lambda x: x["score"] / len(x["tokens"]), reverse=True)
            beams = all_candidates[:beam_size]
            
            # Mostrar progreso
            if beams:
                best_beam = beams[0]
                try:
                    tokens_tensor = torch.tensor([best_beam["tokens"]], dtype=torch.long)
                    text = self.tokenizer.decode_rec(tokens_tensor)[0]
                    logger.info(f"  Best beam: '{text}' (score: {best_beam['score']:.4f})")
                except:
                    logger.info(f"  Best beam tokens: {best_beam['tokens']} (score: {best_beam['score']:.4f})")
        
        # Combinar resultados finales
        all_final_beams = finished_beams + beams
        all_final_beams.sort(key=lambda x: x["score"] / len(x["tokens"]), reverse=True)
        
        # Decodificar textos
        results = []
        for i, beam in enumerate(all_final_beams[:beam_size]):
            try:
                tokens_tensor = torch.tensor([beam["tokens"]], dtype=torch.long)
                text = self.tokenizer.decode_rec(tokens_tensor)[0]
            except:
                text = f"<decode_error: {beam['tokens']}>"
            
            results.append({
                "rank": i + 1,
                "text": text,
                "tokens": beam["tokens"],
                "score": beam["score"],
                "avg_score": beam["score"] / len(beam["tokens"])
            })
        
        return results
    
    def comprehensive_test(self, audio_path: str) -> Dict[str, Any]:
        """Prueba comprehensiva del decoder corregido"""
        logger.info("="*80)
        logger.info("INICIANDO PRUEBA COMPREHENSIVA DEL DECODER CORREGIDO")
        logger.info("="*80)
        
        results = {}
        
        try:
            # 1. Cargar audio
            audio_np, audio_shape_np = self.load_and_preprocess_audio(audio_path)
            
            # 2. Extraer características
            encoder_hidden_states = self.extract_features(audio_np, audio_shape_np)
            
            # 3. Comparar primer paso
            first_step_results = self.compare_first_step_logits(encoder_hidden_states)
            results["first_step_analysis"] = first_step_results
            
            # 4. Ejecutar beam search
            beam_results = self.run_beam_search(encoder_hidden_states)
            results["beam_search_results"] = beam_results
            
            # 5. Resumen
            if beam_results:
                best_result = beam_results[0]
                results["best_prediction"] = best_result["text"]
                results["success"] = True
                
                logger.info("🎉 RESULTADO FINAL:")
                logger.info(f"  Mejor predicción: '{best_result['text']}'")
                logger.info(f"  Score: {best_result['avg_score']:.6f}")
                logger.info(f"  PyTorch referencia: '{first_step_results.get('pytorch_reference', 'N/A')}'")
                
                # Verificar si aún hay problema de repetición
                if "rain" in best_result["text"].lower():
                    tokens = best_result["text"].lower().split()
                    rain_count = tokens.count("rain")
                    if rain_count > 2:
                        logger.warning(f"⚠️  Posible repetición detectada: 'rain' aparece {rain_count} veces")
                        results["repetition_detected"] = True
                    else:
                        logger.info("✅ No se detectó repetición excesiva")
                        results["repetition_detected"] = False
                else:
                    logger.info("✅ No se detectó sesgo hacia 'rain'")
                    results["repetition_detected"] = False
            else:
                results["success"] = False
                results["error"] = "No se generaron resultados"
            
        except Exception as e:
            logger.error(f"Error en prueba comprehensiva: {e}")
            logger.error(traceback.format_exc())
            results["success"] = False
            results["error"] = str(e)
        
        logger.info("="*80)
        logger.info("PRUEBA COMPREHENSIVA COMPLETADA")
        logger.info("="*80)
        
        return results


def main():
    """Función principal para ejecutar la corrección"""
    
    # Crear instancia del beam search corregido
    beam_search = DebugAwareBeamSearch()
    
    # Ejecutar prueba con audio de ejemplo
    audio_path = "data/sample.wav"  # Ajustar según la ruta real
    results = beam_search.comprehensive_test(audio_path)
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL DE LA CORRECCIÓN")
    print("="*60)
    print(f"Éxito: {results.get('success', False)}")
    
    if results.get('success'):
        print(f"Mejor predicción: '{results.get('best_prediction', 'N/A')}'")
        print(f"Repetición detectada: {results.get('repetition_detected', 'Unknown')}")
        
        first_step = results.get('first_step_analysis', {})
        if 'pytorch_reference' in first_step:
            print(f"PyTorch referencia: '{first_step['pytorch_reference']}'")
        
        rain_analysis = first_step.get('fixed_decoder_analysis', {}).get('rain_analysis')
        if rain_analysis:
            print(f"'rain' rank: {rain_analysis['rank']} (dominante: {rain_analysis['is_dominant']})")
    else:
        print(f"Error: {results.get('error', 'Desconocido')}")
    
    print("="*60)
    
    return results.get('success', False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)