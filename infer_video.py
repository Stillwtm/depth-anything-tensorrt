import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptTrtInference
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

    dpt = DptTrtInference(args.engine, args.batch, (height, width), (height, width))

    input_imgs = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_imgs.append(load_frame(frame))

    input_imgs = np.concatenate(input_imgs, axis=0)

    frame_count = 0
    start_time = time.time()

    depths = []
    for i in range(0, input_imgs.shape[0], args.batch):
        # Our implementation only support full batch
        if i + args.batch > input_imgs.shape[0]:
            break

        input_img = input_imgs[i:i+args.batch]
        depths.append(d := dpt(input_img))
        
        frame_count += args.batch
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    average_fps = frame_count / elapsed_time
    print(f"Average FPS: {average_fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.basename(args.video)
    output_path = os.path.join(args.outdir, f'{os.path.splitext(video_name)[0]}_depth.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for depth in depths:
        depth = depth.cpu().numpy().astype(np.uint8)
        for d in depth:
            if args.grayscale:
                depth_colored = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
            else:
                depth_colored = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
            out.write(depth_colored)

    cap.release()
    out.release()

    print(f"Depth video saved to {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation on a video with a TensorRT engine.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth video')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--batch', type=int, default=1, help='Use batch mode for inference')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    args = parser.parse_args()

    run(args)