"""
@author: Hmily
@title: ComfyUI-Light-Tool
@nickname: ComfyUI-Light-Tool
@description: An awesome light tool nodes for ComfyUI.
"""
import sys
import os

from PIL.Image import Resampling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from light_tool_utils import *


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class ResizeImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "resize_method": (["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"], {"default": "LANCZOS"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_img"
    CATEGORY = 'ComfyUI-Light-Tool/image/Resize'
    DESCRIPTION = "Crop an image based on its width and height"

    @staticmethod
    def resize_img(image, width, height, resize_method, mode):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert(mode)
            if width is None and height is None:
                raise ValueError("Either new_width or new_height must be provided.")
            elif width is not None and height is None:
                height = int((width / img.width) * img.height)
            elif height is not None and width is None:
                width = int((height / img.height) * img.width)

            method = {
                "LANCZOS": Image.LANCZOS,
                "BICUBIC": Image.BICUBIC,
                "NEAREST": Image.NEAREST,
                "BILINEAR": Image.BILINEAR
            }

            img_resized = img.resize((width, height), method[resize_method])
            result_img = pil2tensor(img_resized)
            image_list.append(result_img)

        image = torch.cat(image_list, dim=0)
        return (image,)


class ResizeImageV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "base": (["width", "height"], {"default": "width"}),
                "resize_method": (["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"], {"default": "LANCZOS"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_img_v2"
    CATEGORY = 'ComfyUI-Light-Tool/image/Resize'
    DESCRIPTION = "Reduce the image by a fixed width or height"

    @staticmethod
    def resize_img_v2(image, width, height, base, resize_method, mode):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert(mode)

            method = {
                "LANCZOS": Resampling.LANCZOS,
                "BICUBIC": Resampling.BICUBIC,
                "NEAREST": Resampling.NEAREST,
                "BILINEAR": Resampling.BILINEAR
            }

            original_width, original_height = img.size

            if base == 'width':
                w_percent = width / float(original_width)
                new_height = int(float(original_height) * w_percent)
                resized_img = img.resize((width, new_height), method[resize_method])
            else:
                w_percent = height / float(original_height)
                new_width = int(float(original_width) * w_percent)
                resized_img = img.resize((new_width, height), method[resize_method])

            result_img = pil2tensor(resized_img)
            image_list.append(result_img)

        image = torch.cat(image_list, dim=0)
        return (image,)


class ResizeImageByMaxSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_width": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "max_height": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "resize_method": (["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"], {"default": "LANCZOS"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_img_by_max_size"
    CATEGORY = 'ComfyUI-Light-Tool/image/Resize'
    DESCRIPTION = "Resize the image proportionally to ensure its resolution does not exceed the specified maximum size"

    @staticmethod
    def resize_img_by_max_size(image, max_width, max_height, resize_method, mode):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert(mode)

            method = {
                "LANCZOS": Resampling.LANCZOS,
                "BICUBIC": Resampling.BICUBIC,
                "NEAREST": Resampling.NEAREST,
                "BILINEAR": Resampling.BILINEAR
            }

            original_width, original_height = img.size
            ratio = min(max_width / original_width, max_height / original_height)

            if ratio >= 1:
                return (pil2tensor(img),)

            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            resized_img = img.resize((new_width, new_height), method[resize_method])
            result_img = pil2tensor(resized_img)
            image_list.append(result_img)

        image = torch.cat(image_list, dim=0)
        return (image,)


class ResizeImageByMinSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_width": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "min_height": ("INT", {"default": 512, "min": 0, "max": 8192, "display": "number"}),
                "resize_method": (["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"], {"default": "LANCZOS"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_img_by_min_size"
    CATEGORY = 'ComfyUI-Light-Tool/image/Resize'
    DESCRIPTION = ("Resize the image proportionally to ensure its resolution does not fall below the specified minimum "
                   "size")

    @staticmethod
    def resize_img_by_min_size(image, min_width, min_height, resize_method, mode):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert(mode)

            method = {
                "LANCZOS": Resampling.LANCZOS,
                "BICUBIC": Resampling.BICUBIC,
                "NEAREST": Resampling.NEAREST,
                "BILINEAR": Resampling.BILINEAR
            }

            original_width, original_height = img.size
            ratio_w = min_width / original_width
            ratio_h = min_height / original_height
            ratio = max(ratio_w, ratio_h)

            if ratio <= 1:
                return (pil2tensor(img),)

            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            resized_img = img.resize((new_width, new_height), method[resize_method])
            result_img = pil2tensor(resized_img)
            image_list.append(result_img)

        image = torch.cat(image_list, dim=0)
        return (image,)


class ResizeImageByRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0, "display": "number"}),
                "resize_method": (["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"], {"default": "LANCZOS"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_img_by_ratio"
    CATEGORY = 'ComfyUI-Light-Tool/image/Resize'
    DESCRIPTION = "Reduce the image by a fixed width or height"

    @staticmethod
    def resize_img_by_ratio(image, ratio, resize_method, mode):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert(mode)

            method = {
                "LANCZOS": Resampling.LANCZOS,
                "BICUBIC": Resampling.BICUBIC,
                "NEAREST": Resampling.NEAREST,
                "BILINEAR": Resampling.BILINEAR
            }

            original_width, original_height = img.size
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            resized_img = img.resize((new_width, new_height), method[resize_method])
            result_img = pil2tensor(resized_img)
            image_list.append(result_img)

        image = torch.cat(image_list, dim=0)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "Light-Tool: ResizeImage": ResizeImage,
    "Light-Tool: ResizeImageV2": ResizeImageV2,
    "Light-Tool: ResizeImageByRatio": ResizeImageByRatio,
    "Light-Tool: ResizeImageByMaxSize": ResizeImageByMaxSize,
    "Light-Tool: ResizeImageByMinSize": ResizeImageByMinSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: ResizeImage": "Light-Tool: Resize Image",
    "Light-Tool: ResizeImageV2": "Light-Tool: Resize Image V2",
    "Light-Tool: ResizeImageByRatio": "Light-Tool: Resize Image By Ratio",
    "Light-Tool: ResizeImageByMaxSize": "Light-Tool: Resize Image By Max Size",
    "Light-Tool: ResizeImageByMinSize": "Light-Tool: Resize Image By Min Size"
}
