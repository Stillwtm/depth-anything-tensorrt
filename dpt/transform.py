import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DptPreProcess(object):
    def __init__(self, input_size, target_size, img_channel=3, ensure_multiple_of=14, dtype=torch.float32, device='cuda'):
        self._multiple_of = ensure_multiple_of
        self._img_channel = img_channel
        self._device = device
        self._height, self._width = self.get_input_size(input_size[0], input_size[1], target_size, target_size)

        self._transforms = nn.Sequential(
            T.Resize((self._height, self._width)),
            T.ConvertImageDtype(dtype),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self._multiple_of) * self._multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self._multiple_of) * self._multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self._multiple_of) * self._multiple_of).astype(int)

        return y

    def get_input_size(self, orig_height, orig_width, target_height, target_width):
        scale_height = target_height / orig_height
        scale_width = target_width / orig_width

        if scale_width > scale_height:
            scale_height = scale_width
        else:
            scale_width = scale_height

        new_height = self.constrain_to_multiple_of(scale_height * orig_height, min_val=target_height)
        new_width = self.constrain_to_multiple_of(scale_width * orig_width, min_val=target_width)

        return (new_height, new_width)

    def __call__(self, img):
        img = torch.as_tensor(img, device=self._device)
        img = img.reshape(-1, self._img_channel, *img.shape[-2:])
        img = self._transforms(img)
        img = img.reshape(-1, self._img_channel, self._height, self._width)
        return img
    

class DptPostProcess(object):
    def __init__(self, depth_shape, target_size, dtype=torch.float32):
        self._depth_shape = depth_shape
        self._target_size = target_size
        self._dtype = dtype

    def _normalize(self, x):
        """Per channel normalize
        """
        out_min = x.amin((-2, -1), keepdim=True)
        out_max = x.amax((-2, -1), keepdim=True)
        return (x - out_min) / (out_max - out_min) * 255.

    def __call__(self, depth):
        depth = self._normalize(depth.reshape(*self._depth_shape))
        # import ipdb; ipdb.set_trace()
        depth = F.interpolate(depth.unsqueeze(1), size=self._target_size, mode='bilinear', align_corners=False)

        return depth.to(self._dtype).squeeze(1)