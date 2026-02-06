#!/usr/bin/env python3
"""
convert_keras3_to_h5.py

Convert Keras 3 .keras (ZIP) model files to Keras 2 compatible .h5 format.

Keras 3 saves models as ZIP archives containing:
  - metadata.json
  - config.json
  - model.weights.h5 (with Keras 3 weight layout)

Keras 2 (tf.keras) expects .h5 files with a different weight structure.

This script bridges the gap by:
  1. Building the model architecture using cellfinder's tools (Keras 2)
  2. Loading weights from the Keras 3 .h5 file by layer name
  3. Saving as a full Keras 2 .h5 model

Usage:
    python convert_keras3_to_h5.py model.keras
    python convert_keras3_to_h5.py model.keras --output model_converted.h5
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile

import h5py
import numpy as np


def extract_keras3_weights(keras_path: str) -> str:
    """Extract model.weights.h5 from a .keras ZIP archive. Returns temp path."""
    tmp_dir = tempfile.mkdtemp(prefix="keras_convert_")
    weights_path = os.path.join(tmp_dir, "weights.h5")

    with zipfile.ZipFile(keras_path, 'r') as zf:
        with zf.open('model.weights.h5') as src, open(weights_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    return weights_path


def load_keras3_weights(weights_h5_path: str) -> dict:
    """
    Read Keras 3 weights into a dict: {layer_name: [array0, array1, ...]}.

    Keras 3 stores weights as: layers/{layer_name}/vars/{0,1,2,...}
    """
    weights = {}
    with h5py.File(weights_h5_path, 'r') as f:
        layers_group = f['layers']
        for layer_name in layers_group:
            vars_group = layers_group[layer_name].get('vars')
            if vars_group is None or len(vars_group) == 0:
                continue
            # Read variables in numeric order
            indices = sorted(vars_group.keys(), key=lambda x: int(x))
            weights[layer_name] = [np.array(vars_group[idx]) for idx in indices]
    return weights


def build_config_to_weights_mapping(config, k3_weights):
    """
    Build mapping from config layer names (cellfinder-style) to weight file
    layer names (generic Keras 3 sequential names).

    Keras 3 assigns generic names per class type in construction order:
      Conv3D: conv3d, conv3d_1, conv3d_2, ...
      BatchNormalization: batch_normalization, batch_normalization_1, ...
      Dense: dense, dense_1, ...

    The config.json lists layers in the same construction order with
    cellfinder-style names (conv1, conv1_bn, resunit2_block0_conv_a, ...).
    """
    # Map Keras class names to their generic weight-file base names
    class_to_generic = {
        'Conv3D': 'conv3d',
        'BatchNormalization': 'batch_normalization',
        'Dense': 'dense',
    }

    # Track sequential index per class type
    class_counters = {cls: 0 for cls in class_to_generic}
    mapping = {}  # config_name -> weight_file_name

    for layer in config['config']['layers']:
        cls = layer['class_name']
        config_name = layer.get('name', '')
        if cls not in class_to_generic:
            continue

        idx = class_counters[cls]
        if idx == 0:
            generic_name = class_to_generic[cls]
        else:
            generic_name = f"{class_to_generic[cls]}_{idx}"
        class_counters[cls] += 1

        # Only include if this generic name actually has weights
        if generic_name in k3_weights:
            mapping[config_name] = generic_name

    return mapping


def convert_model(keras_path: str, output_path: str):
    """Convert a Keras 3 .keras model to Keras 2 .h5 format."""
    print(f"Input:  {keras_path}")
    print(f"Output: {output_path}")

    # Step 1: Extract weights from ZIP and read config
    print("Extracting weights from .keras archive...")
    weights_path = extract_keras3_weights(keras_path)
    k3_weights = load_keras3_weights(weights_path)
    print(f"  Found {len(k3_weights)} layers with weights in weight file")

    with zipfile.ZipFile(keras_path, 'r') as zf:
        config = json.loads(zf.read('config.json'))

    # Step 2: Build mapping from config names to weight-file names
    print("Building config-name to weight-file-name mapping...")
    name_mapping = build_config_to_weights_mapping(config, k3_weights)
    print(f"  Mapped {len(name_mapping)} layers")

    # Step 3: Build the model architecture using cellfinder's tools
    print("Building ResNet50 model architecture (Keras 2)...")

    from cellfinder.core.classify.tools import get_model

    model = get_model(
        existing_model=None,
        model_weights=None,
        network_depth="50-layer",
        learning_rate=0.0001,
        continue_training=False,
    )
    print(f"  Model: {model.name}, {len(model.layers)} layers")

    # Step 4: Set weights on Keras 2 model layers using the mapping
    # config names (cellfinder-style) match Keras 2 layer names
    print("Loading weights into Keras 2 model...")

    k2_layers_by_name = {layer.name: layer for layer in model.layers}

    matched = 0
    mismatched = 0
    skipped = 0

    for config_name, weight_name in name_mapping.items():
        if config_name not in k2_layers_by_name:
            print(f"  SKIP: config '{config_name}' not found in Keras 2 model")
            skipped += 1
            continue

        k2_layer = k2_layers_by_name[config_name]
        k2_w = k2_layer.get_weights()
        k3_w = k3_weights[weight_name]

        if len(k3_w) != len(k2_w):
            print(f"  MISMATCH: {config_name}: "
                  f"K2 has {len(k2_w)} arrays, K3 ({weight_name}) has {len(k3_w)} arrays")
            mismatched += 1
            continue

        shapes_ok = all(k3_w[i].shape == k2_w[i].shape for i in range(len(k3_w)))
        if not shapes_ok:
            print(f"  SHAPE MISMATCH: {config_name} ({weight_name}):")
            for i in range(len(k3_w)):
                print(f"    [{i}] K2: {k2_w[i].shape} vs K3: {k3_w[i].shape}")
            mismatched += 1
            continue

        k2_layer.set_weights(k3_w)
        matched += 1

    print(f"  Matched: {matched}, Mismatched: {mismatched}, Skipped: {skipped}")

    # Step 4: Save as Keras 2 .h5
    print(f"Saving as Keras 2 .h5...")
    model.save(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Cleanup temp files
    os.unlink(weights_path)
    os.rmdir(os.path.dirname(weights_path))

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras 3 .keras models to Keras 2 .h5 format"
    )
    parser.add_argument('input', help='Path to .keras model file')
    parser.add_argument('--output', '-o', help='Output .h5 path (default: {input}_keras2.h5)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    if not zipfile.is_zipfile(args.input):
        print(f"ERROR: Not a valid .keras (ZIP) file: {args.input}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.input)[0]
        output_path = f"{base}_keras2.h5"

    success = convert_model(args.input, output_path)
    if success:
        print("\nConversion complete!")
    else:
        print("\nConversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
