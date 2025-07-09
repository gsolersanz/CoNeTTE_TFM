#!/usr/bin/env python3

import onnx
import sys
import os
from pathlib import Path

# Check decoder inputs
base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
decoder_path = str(base_dir / "conette_t5/dec_no_cache/model.onnx")

print(f"Checking decoder at: {decoder_path}")
print(f"Exists: {os.path.exists(decoder_path)}")

if os.path.exists(decoder_path):
    model = onnx.load(decoder_path)
    print('T5 DECODER INPUTS:')
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        dtype_map = {1: 'float32', 6: 'int32', 7: 'int64', 9: 'bool'}
        dtype_name = dtype_map.get(inp.type.tensor_type.elem_type, f'unknown({inp.type.tensor_type.elem_type})')
        print(f'  {inp.name}: {shape} ({dtype_name})')
    
    print('\nT5 DECODER OUTPUTS:')
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
        dtype_map = {1: 'float32', 6: 'int32', 7: 'int64', 9: 'bool'}
        dtype_name = dtype_map.get(out.type.tensor_type.elem_type, f'unknown({out.type.tensor_type.elem_type})')
        print(f'  {out.name}: {shape} ({dtype_name})')
else:
    print("T5 Decoder not found!")