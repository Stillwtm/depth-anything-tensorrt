import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptInference


def load_image(filepath):
    img = Image.open(filepath)  # H, W, C
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.uint8)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_img = load_image(args.img)

    dpt = DptInference(args.engine, 1, input_img.shape[2:], (512, 512))
    depth = dpt(input_img)

    # Save depth map
    img_name = os.path.basename(args.img)
    output_path = f'{args.outdir}/{os.path.splitext(img_name)[0]}_depth.png'
    depth = depth.squeeze().cpu().numpy().astype(np.uint8)
    if args.grayscale:
        cv2.imwrite(output_path, depth)
    else:
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, colored_depth)

    print(f"Depth saved to {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    args = parser.parse_args()

    run(args)