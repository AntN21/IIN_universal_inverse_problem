from typing import Any
from skimage.restoration import denoise_tv_chambolle
import torch
import cv2 as cv
import numpy as np

class TVDenoiser():
    def __init__(self, weight = 1.) -> None:
        self.model = denoise_tv_chambolle
        self.weight = weight

    
    def __call__(self, x, weight = None) -> Any:
        if weight is None:
            weight = self.weight

        if torch.is_tensor(x):
            x = x.numpy()
        
        return torch.tensor(self.model(x, weight))


class CVDenoiser():
    def __init__(self) -> None:

        self.model = cv.fastNlMeansDenoisingColored
    
    def __call__(self, x) -> Any:

        if torch.is_tensor(x):
            # print('x0',x.shape)
            x = torch.squeeze(x, 0)
            # x = x.squeeze()
            x_new = torch.transpose(x,-3,-2)
            x_new = torch.transpose(x_new,-2,-1)
            img = np.array(255*x_new.numpy(), dtype=np.uint8)
            # print(img)
            dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

            res = torch.tensor(dst).transpose(-2,-1).transpose(-2,-3) / 255.
        # print(res)
        return res.unsqueeze(0)