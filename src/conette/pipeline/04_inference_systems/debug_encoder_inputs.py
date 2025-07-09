#!/usr/bin/env python3

import onnx
import sys
import os
from pathlib import Path

# Check encoder inputs in working models
base_dir = Path(__file__).parent.parent.parent  # Volver a conette/
encoder_path = str(base_dir / "onnx_models_full/conette_encoder.onnx")

print(f"Checking encoder at: {encoder_path}")
print(f"Exists: {os.path.exists(encoder_path)}")

if os.path.exists(encoder_path):
    model = onnx.load(encoder_path)
    print('ENCODER INPUTS:')
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        dtype_map = {1: 'float32', 6: 'int32', 7: 'int64', 9: 'bool'}
        dtype_name = dtype_map.get(inp.type.tensor_type.elem_type, f'unknown({inp.type.tensor_type.elem_type})')
        print(f'  {inp.name}: {shape} ({dtype_name})')
else:
    print("Encoder not found!")

# Also check working system directory
working_encoder = "/workspace/conette/onnx_models_full/conette_encoder.onnx"
print(f"\nChecking working encoder at: {working_encoder}")
print(f"Exists: {os.path.exists(working_encoder)}")

if os.path.exists(working_encoder):
    model = onnx.load(working_encoder)
    print('WORKING ENCODER INPUTS:')
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        dtype_map = {1: 'float32', 6: 'int32', 7: 'int64', 9: 'bool'}
        dtype_name = dtype_map.get(inp.type.tensor_type.elem_type, f'unknown({inp.type.tensor_type.elem_type})')
        print(f'  {inp.name}: {shape} ({dtype_name})')