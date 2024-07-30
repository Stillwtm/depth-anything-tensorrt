import argparse
import os
import cv2
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
    
    cap = cv2.VideoCapture(args.webcam)  # 使用摄像头
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(args.outdir, 'webcam_depth.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    dpt = DptTrtInference(args.engine, 1, (height, width), (height, width))

    frame_count = 0
    start_time = time.time()

    while True:
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

        # 计算实时 FPS
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time
        print(f"Current FPS: {current_fps:.2f}", end='\r')

        # 显示实时视频
        cv2.imshow('Depth Video', depth_colored)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    average_fps = frame_count / total_elapsed_time

    print(f"\nDepth video saved to {output_path}.")
    print(f"Average FPS: {average_fps:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation on webcam video with a TensorRT engine.')
    parser.add_argument('--webcam', type=str, default='0', help='Webcam id')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth video')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    args = parser.parse_args()

    run(args)