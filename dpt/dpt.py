
import torch
import numpy as np
import tensorrt as trt
from .transform import DptPreProcess, DptPostProcess


class DptTrtInference:
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
        
        self._context = self._engine.create_execution_context()
        
        # Get the shape and data type of the input and output
        # Batch (first) dimension is dummy here.
        self._input_shape = self._engine.get_tensor_shape('input')
        self._output_shape = self._engine.get_tensor_shape('output')
        self._input_shape[0] = self._batch_size
        self._output_shape[0] = self._batch_size
        input_dtype = trt.nptype(self._engine.get_tensor_dtype('input'))
        output_dtype = trt.nptype(self._engine.get_tensor_dtype('output'))
        
        self._context.set_input_shape('input', self._input_shape)

        # Allocate memory
        self._d_input = torch.empty(tuple(self._input_shape), dtype=torch.from_numpy(np.array([], dtype=input_dtype)).dtype, device='cuda').view(-1)
        self._d_output = torch.empty(tuple(self._output_shape), dtype=torch.from_numpy(np.array([], dtype=output_dtype)).dtype, device='cuda').view(-1)

        # Create a CUDA stream for asynchronous processing
        self._stream = torch.cuda.Stream()

    @torch.no_grad()
    def __call__(self, img):
        img = self._pre_process(img)

        # Copy input data to GPU memory
        self._d_input.copy_(img.contiguous().view(-1))
        torch.cuda.current_stream().synchronize()

        # Set input and output bindings
        self._context.set_tensor_address('input', self._d_input.data_ptr())
        self._context.set_tensor_address('output', self._d_output.data_ptr())

        self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        depth = self._post_process(self._d_output)

        return depth
