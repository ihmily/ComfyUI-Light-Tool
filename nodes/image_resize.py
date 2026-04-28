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


class AspectRatioPadder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_ratio": (["1:1", "9:16", "16:9", "4:3", "3:2", "2:3"], {"default": "1:1"}),
                "custom_ratio": ("STRING", {"default": ""}),
                "base_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "display": "number"}),
                "color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "use_hex": ("BOOLEAN", {"default": True}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pad_image"
    CATEGORY = 'ComfyUI-Light-Tool/image/Resize'
    DESCRIPTION = "Resize image to fit target ratio and pad with white background."

    @staticmethod
    def pad_image(image, target_ratio, custom_ratio, base_size, color_hex, use_hex, R, G, B):
        ratio_map = {
            "1:1": 1.0,
            "9:16": 9.0 / 16.0,
            "16:9": 16.0 / 9.0,
            "4:3": 4.0 / 3.0,
            "3:2": 3.0 / 2.0,
            "2:3": 2.0 / 3.0,
            "None": None
        }

        target_w_h_ratio = ratio_map.get(custom_ratio or target_ratio)
        if not target_w_h_ratio:
            out_images = torch.cat([image], dim=0)
            return (out_images,)

        image_list = []

        for img in image:
            pil_img = tensor2pil(img)
            orig_w, orig_h = pil_img.size
            orig_ratio = orig_w / orig_h

            if target_w_h_ratio >= 1.0:  # horizontal image (16:9, 4:3, 3:2)
                target_w = base_size
                target_h = int(base_size / target_w_h_ratio)
            else:  # vertical image or Square Map (9:16, 2:3, 1:1)
                target_h = base_size
                target_w = int(base_size * target_w_h_ratio)

            target_w = (target_w // 8) * 8
            target_h = (target_h // 8) * 8

            if use_hex:
                rgb_background_color = hex_to_rgb(color_hex)
            else:
                rgb_background_color = (R, G, B)
            new_image = Image.new("RGB", (target_w, target_h), rgb_background_color)

            if orig_ratio > target_w_h_ratio:
                scale_ratio = target_w / orig_w
            else:
                scale_ratio = target_h / orig_h

            resize_w = int(orig_w * scale_ratio)
            resize_h = int(orig_h * scale_ratio)

            resize_w = min(resize_w, target_w)
            resize_h = min(resize_h, target_h)

            resized_img = pil_img.resize((resize_w, resize_h), resample=Image.Resampling.LANCZOS)

            paste_x = (target_w - resize_w) // 2
            paste_y = (target_h - resize_h) // 2

            new_image.paste(resized_img, (paste_x, paste_y))

            result_tensor = pil2tensor(new_image)
            image_list.append(result_tensor)

        out_images = torch.cat(image_list, dim=0)
        return (out_images,)


NODE_CLASS_MAPPINGS = {
    "Light-Tool: ResizeImage": ResizeImage,
    "Light-Tool: ResizeImageV2": ResizeImageV2,
    "Light-Tool: ResizeImageByRatio": ResizeImageByRatio,
    "Light-Tool: ResizeImageByMaxSize": ResizeImageByMaxSize,
    "Light-Tool: ResizeImageByMinSize": ResizeImageByMinSize,
    "Light-Tool: AspectRatioPadder": AspectRatioPadder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: ResizeImage": "Light-Tool: Resize Image",
    "Light-Tool: ResizeImageV2": "Light-Tool: Resize Image V2",
    "Light-Tool: ResizeImageByRatio": "Light-Tool: Resize Image By Ratio",
    "Light-Tool: ResizeImageByMaxSize": "Light-Tool: Resize Image By Max Size",
    "Light-Tool: ResizeImageByMinSize": "Light-Tool: Resize Image By Min Size",
    "Light-Tool: AspectRatioPadder": "Light-Tool: Ratio Padder"
}
