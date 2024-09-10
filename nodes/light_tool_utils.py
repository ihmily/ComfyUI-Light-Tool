"""
@author: Hmily
@title: ComfyUI-Light-Tool
@nickname: ComfyUI-Light-Tool
@description: An awesome light image processing tool nodes for ComfyUI.
"""
from typing import Union, List
import cv2
import numpy as np
import torch
from PIL import Image


def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def RGB2RGBA(image: Image, mask: Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))


def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    if len(tensor.shape) == 3:
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]


def np2pil(np_image: np.ndarray, mode: str = 'RGB') -> Image:
    return Image.fromarray((np_image * 255).astype(np.uint8), mode)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


def to_gray(image, method="luminosity"):
    if method == "luminosity":
        # Standard weighted average method (0.299*R + 0.587*G + 0.114*B)
        return image.convert("L")
    else:
        image = image.convert("RGB")
        width, height = image.size
        gray_image = Image.new("L", (width, height))

        for x in range(width):
            for y in range(height):
                r, g, b = image.getpixel((x, y))

                if method == "average":
                    gray = (r + g + b) / 3
                elif method == "max":
                    gray = max(r, g, b)
                elif method == "min":
                    gray = min(r, g, b)
                elif method == "custom":
                    gray = 0.3 * r + 0.5 * g + 0.2 * b
                else:
                    raise ValueError("Unsupported grayscale conversion method.")

                gray_image.putpixel((x, y), int(gray))

        return gray_image
