
import torch
import numpy as np
import tensorrt as trt
from .transform import DptPreProcess, DptPostProcess


class DptInference:
    def __init__(self, engine_path, batch_size, img_size, depth_size, dpt_size=518):
        self._engine_path = engine_path
        self._batch_size = batch_size
        self._device = torch.device('cuda')
        self._prepare()
        self._pre_process = DptPreProcess(img_size, dpt_size, device=self._device)
        self._post_process = DptPostProcess(self._output_shape, depth_size)

    def _prepare(self):
        # Prepare engine
        logger = trt.Logger(trt.Logger.WARNING)
        with trt.Runtime(logger) as runtime:
            with open(self._engine_path, 'rb') as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())

        # Get the shape and data type of the input and output
        # Batch (first) dimension is dummy here.
        self._input_shape = self._engine.get_tensor_shape('image')
        self._output_shape = self._engine.get_tensor_shape('depth')
        self._input_shape[0] = self._batch_size
        self._output_shape[0] = self._batch_size
        input_dtype = trt.nptype(self._engine.get_tensor_dtype('image'))
        output_dtype = trt.nptype(self._engine.get_tensor_dtype('depth'))

        # Allocate persistent memory
        self._d_input = torch.empty(tuple(self._input_shape), dtype=torch.from_numpy(np.array([], dtype=input_dtype)).dtype, device='cuda').view(-1)
        self._d_output = torch.empty(tuple(self._output_shape), dtype=torch.from_numpy(np.array([], dtype=output_dtype)).dtype, device='cuda').view(-1)

    def __call__(self, img):
        img = self._pre_process(img)
        
        # Create a CUDA stream for asynchronous processing
        stream = torch.cuda.Stream()

        with self._engine.create_execution_context() as context:
            # Set input and output bindings
            context.set_tensor_address('image', self._d_input.data_ptr())
            context.set_tensor_address('depth', self._d_output.data_ptr())

            # Copy input data to GPU memory
            self._d_input.copy_(img.view(-1))

            context.execute_async_v3(stream.cuda_stream)
            # stream.synchronize()

            # Create a PyTorch tensor that references the GPU memory
            depth = self._d_output

        depth = self._post_process(depth)

        return depth
