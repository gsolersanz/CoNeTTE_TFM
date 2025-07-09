#!/usr/bin/env python3

import onnx
import numpy as np
import onnxruntime as ort

# Ver inputs exactos del decoder
decoder_model = onnx.load('../06_models/t5_models/dec_no_cache/model.onnx')
print("DECODER INPUTS:")
for inp in decoder_model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
    dtype_map = {1: 'float32', 6: 'int32', 7: 'int64', 9: 'bool'}
    dtype_name = dtype_map.get(inp.type.tensor_type.elem_type, f'unknown({inp.type.tensor_type.elem_type})')
    print(f"  {inp.name}: {shape} ({dtype_name})")

# Probar con valores dummy
print("\nPROBANDO DECODER:")
session = ort.InferenceSession('../06_models/t5_models/dec_no_cache/model.onnx')

# Test básico
frame_embs = np.random.randn(10, 1, 256).astype(np.float32)
frame_embs_pad_mask = np.ones((10, 1), dtype=np.int64)

# Probar diferentes tipos para caps_in
for desc, caps_in in [
    ("caps_in int64", np.array([[1], [2]], dtype=np.int64)),
    ("caps_in bool", np.array([[True], [True]], dtype=bool)),
    ("caps_in int32", np.array([[1], [2]], dtype=np.int32)),
]:
    try:
        inputs = {
            'frame_embs': frame_embs,
            'frame_embs_pad_mask': frame_embs_pad_mask,
            'caps_in': caps_in
        }
        outputs = session.run(None, inputs)
        print(f"✅ {desc}: SUCCESS! Output shape: {outputs[0].shape}")
        break
    except Exception as e:
        print(f"❌ {desc}: {str(e)[:100]}")