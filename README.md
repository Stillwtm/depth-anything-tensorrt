## Depth-Anything-V2 TensorRT Python

Use TensorRT to accelerate the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) model for monocular depth estimation.

## Installation

Clone this repository and submodules:
```bash
git clone https://github.com/Stillwtm/depth-anything-tensorrt.git
git submodule init && git submodule update
```

Install dependencies:
```bash
pip install tensorrt==10.2.0.post1
```

## Model Preparation

### Download Checkpoints

Download the Depth-Anything-V2 checkpoints from [official repository](https://github.com/DepthAnything/Depth-Anything-V2), and put them under the `checkpoints` folder.

### Modify the Model

Replace the `third_party/depth_anything_v2/depth_anything_v2/dpt.py` file with the `tools/dpt.py`. In `tools/dpt.py`, we remove the `squeeze` operation in the `forward` function, which will affect the inference performance of TensorRT models.

### Convert to ONNX

```bash
python tools/export_onnx.py --checkpoint <path to checkpoint> --onnx <path to save onnx model> --input_size <dpt input size> --encoder <dpt encoder> [--dynamic_batch]
```

### Convert ONNX to TensorRT

```bash
python onnx2trt.py --onnx <path to onnx model> --engine <path to save trt engine> [--fp16]
```

You can also enable dynamic batch size for TensorRT engine (If you want to use dynamic batch size here, also remember to enable it in the previous ONNX model conversion step):

```bash
python onnx2trt.py --onnx <path to onnx model> --engine <path to save trt engine> [--fp16] --dynamic_batch --min_batch <minimum batch size> --max_batch <maximum batch size> --opt_batch <optimum batch size>
```

Try to decrease `max_batch` if you encounter a failure (possibly due to OOM error).

After converting the model to TensorRT, you can use the `engine` file for inference.

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