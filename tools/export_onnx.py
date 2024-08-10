import argparse
import cv2
import glob
import matplotlib as plt
import numpy as np
import os
import torch
import torch.onnx

from third_party.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--dynamic_batch', action='store_true', default=False, help='Export the model with dynamic axes')
    args = parser.parse_args()
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to('cpu').eval()

    dummy_input = torch.ones((args.batch, 3, args.input_size, args.input_size))
    example_output = depth_anything.forward(dummy_input)
    dynamic_axes = {'input': {0: 'batch_size'}} if args.dynamic_batch else None

    torch.onnx.export(
        depth_anything,
        dummy_input,
        args.onnx,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    print(f"Model exported to {args.onnx}")

if __name__ == "__main__":
    main()
