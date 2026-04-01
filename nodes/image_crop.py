"""
@author: Hmily
@title: ComfyUI-Light-Tool
@nickname: ComfyUI-Light-Tool
@description: An awesome light tool nodes for ComfyUI.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from light_tool_utils import *


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class CropImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left_margin": ("INT", {"default": 100, "min": 0, "display": "number"}),
                "top_margin": ("INT", {"default": 100, "min": 0, "display": "number"}),
                "right_margin": ("INT", {"default": 100, "min": 0, "display": "number"}),
                "bottom_margin": ("INT", {"default": 100, "min": 0, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_img"
    CATEGORY = 'ComfyUI-Light-Tool/image/Crop'
    DESCRIPTION = "Crop an image based on its left/top/right/bottom margin"

    @staticmethod
    def crop_img(image, left_margin, top_margin, right_margin, bottom_margin):
        img = tensor2pil(image)
        width, height = img.size

        left = left_margin
        top = top_margin
        right = width - right_margin
        bottom = height - bottom_margin

        if left >= right or top >= bottom:
            raise ValueError("Light-Tool: Margin settings result in an invalid cropping region. "
                             "Please check if the left/right or top/bottom margins are too large.")

        cropped_img = img.crop((left, top, right, bottom))
        result_img = pil2tensor(cropped_img)
        return (result_img,)


class SafeImageCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple_of": (["8", "16", "32", "64", "128", "256", "512", "1024"], {"default": "64"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "safe_crop"
    CATEGORY = "ComfyUI-Light-Tool/image/Crop"
    DESCRIPTION = ("Crops image from center to the nearest multiple (e.g., 64) to "
                   "ensure compatibility with VAE/Latent space.")

    def safe_crop(self, image, multiple_of):
        multiple_of = int(multiple_of)

        height = image.shape[1]
        width = image.shape[2]

        target_height = (height // multiple_of) * multiple_of
        target_width = (width // multiple_of) * multiple_of

        if target_height == 0 or target_width == 0:
            print(f"Warning: Image size ({width}x{height}) is too small for multiple of {multiple_of}.")
            return (image,)

        y_start = (height - target_height) // 2
        x_start = (width - target_width) // 2

        cropped_image = image[:, y_start:y_start + target_height, x_start:x_start + target_width, :]

        return (cropped_image,)


NODE_CLASS_MAPPINGS = {
    "Light-Tool: CropImage": CropImage,
    "Light-Tool: SafeImageCrop": SafeImageCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: CropImage": "Light-Tool: Crop Image",
    "Light-Tool: SafeImageCrop": "Light-Tool: Safe Image Crop"
}
