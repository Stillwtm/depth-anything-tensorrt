import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptInference
import time

def load_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # H, W, C
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.uint8)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.basename(args.video)
    output_path = os.path.join(args.outdir, f'{os.path.splitext(video_name)[0]}_depth.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    dpt = DptInference(args.engine, 1, (height, width), (height, width))

    frame_count = 0
    start_time = time.time()

    for _ in range(20):
    # while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_img = load_frame(frame)
        depth = dpt(input_img)

        depth = depth.squeeze().cpu().numpy().astype(np.uint8)
        if args.grayscale:
            depth_colored = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        else:
            depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        out.write(depth_colored)

        frame_count += 1

    cap.release()
    out.release()

    end_time = time.time()
    elapsed_time = end_time - start_time
    average_fps = frame_count / elapsed_time

    print(f"Depth video saved to {output_path}.")
    print(f"Average FPS: {average_fps:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation on a video with a TensorRT engine.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth video')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    args = parser.parse_args()

    run(args)