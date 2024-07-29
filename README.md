## Depth-Anything-V2 TensorRT Python

Use TensorRT to accelerate the Depth-Anything-V2 model for monocular depth estimation.

## Installation

```bash
pip install tensorrt==10.2.0.post1
```

## Model Preparation

Follow [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) to get onnx models of Depth-Anything-V2. Then convert them to TensorRT engine.

```bash
python onnx2trt.py --onnx <path to onnx model> --engine <path to save trt engine> [--fp16]
```

Try to decrease the `MAX_BATCH` if you encounter a failure (maybe due to OOM error).

## Inference

For a single image, use:

```bash
python infer.py --img <path to image> --engine <path to trt engine> [--grayscale]
```

For a video, use:

```bash
python infer_video.py --video <path to video> --engine <path to trt engine> [--batch <batch size>] [--grayscale]
```

You can also use a webcam for real-time inference:

```bash
python infer_webcam.py --webcam <path to video> --engine <path to trt engine> [--grayscale]
```