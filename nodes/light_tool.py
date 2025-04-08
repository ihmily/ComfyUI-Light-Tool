"""
@author: Hmily
@title: ComfyUI-Light-Tool
@nickname: ComfyUI-Light-Tool
@description: An awesome light image processing tool nodes for ComfyUI.
"""
import sys
import os
import io
import hashlib
import time
import uuid

import numpy as np
from PIL import ImageSequence, ImageOps
from typing import Any, Tuple
from torchvision.transforms import functional
import folder_paths
import node_helpers
from oss_tool import oss_upload
from upscale import UpscaleMode, upscale_image
from scale import ScaleMode, scale_image


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from light_tool_utils import *


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class LoadImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "keep_alpha_channel": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = 'ComfyUI-Light-Tool/image/LoadImage'
    DESCRIPTION = "Load image"

    @staticmethod
    def load_image(image, keep_alpha_channel):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda pixel: pixel * (1 / 255))

            has_alpha = "A" in i.getbands()
            if has_alpha and keep_alpha_channel:
                image = i.convert("RGBA")
            else:
                image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None, ]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        return output_image, output_mask

    @classmethod
    def IS_CHANGED(cls, image, keep_alpha_channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image, keep_alpha_channel):
        if not folder_paths.exists_annotated_filepath(image):
            return "LoadImage(Light-Tool): Invalid image file: {}".format(image)

        return True


class LoadImageFromURL:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://www.comfy.org/images/ComfyUI_00000.png", "multiline": True}),
                "keep_alpha_channel": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image_from_url"
    CATEGORY = "ComfyUI-Light-Tool/image/LoadImage"
    DESCRIPTION = "Load image From URL"

    @staticmethod
    def load_image_from_url(url, keep_alpha_channel):
        url_list = url.replace('，', ',').split(',')
        image_list = []
        for url in url_list:
            response = httpx.get(url.strip())
            if response.status_code == 200:
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))
                image = ImageOps.exif_transpose(image)
                has_alpha = "A" in image.getbands()
                if has_alpha and keep_alpha_channel:
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")
                image_list.append(pil2tensor(image))
            else:
                print(f"LoadImageFromURL(Light-Tool): Failed to retrieve image, status code: {response.status_code}")

        image = torch.cat(image_list, dim=0)
        return (image,)


class LoadImagesFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "please input your image dir path"}),
            },
            "optional": {
                "image_load_num": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "keep_alpha_channel": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE PATH")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load_images"
    CATEGORY = 'ComfyUI-Light-Tool/image/LoadImage'
    DESCRIPTION = "Load image From image directory"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    @staticmethod
    def load_images(directory: str, image_load_num: int = 0, start_index: int = 0, keep_alpha_channel: bool = False,
                    load_always: bool = False) -> Tuple[List, List, List]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        file_paths = [os.path.join(directory, x) for x in dir_files]

        from itertools import islice
        file_paths = list(
            islice(file_paths, start_index, None if image_load_num == 0 else start_index + image_load_num))

        images, masks = [], []
        for image_path in file_paths:
            with Image.open(image_path) as i:
                i = ImageOps.exif_transpose(i)
                has_alpha = "A" in i.getbands()
                if has_alpha and keep_alpha_channel:
                    image = i.convert("RGBA")
                else:
                    image = i.convert("RGB")

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None, ]

                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32)

                images.append(image)
                masks.append(mask)

        return images, masks, [str(image_path) for image_path in file_paths]


class ImageMaskApply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Extract the transparent image using a mask to separate the subject from the background"

    @staticmethod
    def run(image, mask, invert):
        image_list = []
        for _image, _mask in zip(image, mask):
            image_pil = tensor2pil(_image).convert('RGB')
            mask_pil = tensor2pil(_mask).convert('L')
            image_size = image_pil.size
            mask_size = mask_pil.size
            if mask_pil.size != image_pil.size:
                raise ValueError(f"ImageMaskApply(Light-Tool): Images must have the same size. "
                                 f"{image_size}and{mask_size} is not match")
            if invert:
                inverted_mask = invert_mask(mask_pil)
                image_pil = rgb2rgba(image_pil, inverted_mask)
            else:
                image_pil = rgb2rgba(image_pil, mask_pil)
            image = pil2tensor(image_pil)
            image_list.append(image)
        image = torch.cat(image_list, dim=0)
        return (image,)


class MaskToImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mask_to_image"
    CATEGORY = 'ComfyUI-Light-Tool/image/mask'
    DESCRIPTION = "Convert mask to image"

    @staticmethod
    def mask_to_image(mask):
        image_list = []
        for _mask in mask:
            result = _mask.reshape((-1, 1, _mask.shape[-2], _mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            image_list.append(result)
        image = torch.cat(image_list, dim=0)
        return (image,)


class ImageToMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["red", "green", "blue", "alpha"], {"default": "red"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_to_mask"
    CATEGORY = 'ComfyUI-Light-Tool/image/mask'
    DESCRIPTION = "Convert image to mask"

    @staticmethod
    def image_to_mask(image, channel):
        mask_list = []
        for img in image:
            channels = ["red", "green", "blue", "alpha"]
            if channel == "alpha":
                img = np.array(tensor2pil(img).convert("RGBA"))
            else:
                img = np.array(tensor2pil(img).convert("RGB"))
            mask = np2tensor(img[:, :, channels.index(channel)])
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim=0)
        return (mask,)


class MaskImageToTransparent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mask2Transparent"
    CATEGORY = 'ComfyUI-Light-Tool/image/mask'
    DESCRIPTION = "Convert the non-masked areas of an image to transparency"

    @staticmethod
    def mask2Transparent(image):
        image_list = []
        for img in image:
            gray_image = tensor2pil(img).convert("L")
            gray_image_with_alpha = Image.new('RGBA', gray_image.size, (255, 255, 255, 0))
            gray_image_with_alpha.paste(gray_image, mask=gray_image)
            transparent_image = pil2tensor(gray_image_with_alpha)
            image_list.append(transparent_image)
        transparent_image = torch.cat(image_list, dim=0)
        return (transparent_image,)


class BoundingBoxCropping:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_crop"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageCrop'
    DESCRIPTION = "Cut out an image based on the smallest bounding rectangle of the transparent object within it"

    @staticmethod
    def image_crop(image):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert("RGBA")
            img = np.array(img)
            alpha_channel = img[:, :, 3]
            bbox = cv2.boundingRect(alpha_channel)
            x, y, w, h = bbox
            cropped_img_array = img[y:y + h, x:x + w]
            cropped_img_array = cv2.cvtColor(cropped_img_array, cv2.COLOR_RGB2RGBA)
            crop_image = np2tensor(cropped_img_array)
            image_list.append(crop_image)
        crop_image = torch.cat(image_list, dim=0)
        return (crop_image,)


class MaskBoundingBoxCropping:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mask_image_crop"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageCrop'
    DESCRIPTION = "Cut out an mask image based on the smallest bounding rectangle of the transparent object within it"

    @staticmethod
    def mask_image_crop(image):
        image_list = []
        for img in image:
            gray_image = tensor2pil(img).convert("L")

            gray_image_with_alpha = Image.new('RGBA', gray_image.size, (255, 255, 255, 0))
            gray_image_with_alpha.paste(gray_image, mask=gray_image)

            img_array = np.array(gray_image_with_alpha)
            if img_array.shape[2] != 4:
                raise ValueError("MaskBoundingBoxCropping(Light-Tool): Input image must have an alpha channel")
            alpha_channel = img_array[:, :, 3]
            bbox = cv2.boundingRect(alpha_channel)
            x, y, w, h = bbox
            img_array = img_array[y:y + h, x:x + w]

            cropped_image = Image.fromarray(img_array, 'RGBA')
            black_background = Image.new('RGBA', cropped_image.size, (0, 0, 0, 255))
            cropped_image = Image.alpha_composite(black_background, cropped_image)
            crop_image = pil2tensor(cropped_image)

            image_list.append(crop_image)
        crop_image = torch.cat(image_list, dim=0)
        return (crop_image,)


class InvertMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "invert_mask"
    CATEGORY = 'ComfyUI-Light-Tool/image/mask'
    DESCRIPTION = "Invert the colors between the masked and unmasked regions of an image"

    @staticmethod
    def invert_mask(image):
        inverted_mask_list = []
        for img in image:
            mask_image = tensor2pil(img).convert('L')
            pixels = mask_image.load()
            width, height = mask_image.size
            for y in range(height):
                for x in range(width):
                    current_value = pixels[x, y]
                    new_value = 255 - current_value
                    pixels[x, y] = new_value
            inverted_mask = pil2tensor(mask_image)
            inverted_mask_list.append(inverted_mask)
        inverted_mask = torch.cat(inverted_mask_list, dim=0)
        return (inverted_mask,)


class AddBackground:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "use_hex": ("BOOLEAN", {"default": True}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_background"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Add solid color background to transparent image"

    @staticmethod
    def add_background(image, color_hex, use_hex, R, G, B):
        if use_hex and not (color_hex.startswith("#") and len(color_hex) == 7):
            raise ValueError("AddBackground(Light-Tool): Invalid hexadecimal color value")
        image_list = []
        for img_tensor in image:
            img = tensor2pil(img_tensor)

            if use_hex:
                rgb_background_color = hex_to_rgb(color_hex)
            else:
                rgb_background_color = (R, G, B)
            background = Image.new("RGB", img.size, rgb_background_color)
            img_array = np.array(img)[:, :, :3]
            alpha_array = np.array(img)[:, :, 3] / 255.0
            bg_array = np.array(background)
            result_array = (img_array * alpha_array[:, :, np.newaxis] + bg_array * (
                    1 - alpha_array[:, :, np.newaxis])).astype(np.uint8)

            result_img = Image.fromarray(result_array)
            new_image = pil2tensor(result_img)
            image_list.append(new_image)

        new_image = torch.cat(image_list, dim=0)
        return (new_image,)


class ImageOverlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "origin_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "overlay_mask": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine_images"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Overlay one image on top of another to create a composite image"

    @staticmethod
    def combine_images(origin_image: torch.Tensor, overlay_image: torch.Tensor, overlay_mask: torch.Tensor):
        image_list = []
        for img in origin_image:
            image = tensor2pil(overlay_image).convert('RGB')
            bg = tensor2pil(img).convert('RGB')
            alpha = tensor2pil(overlay_mask).convert('L')

            if image.size != bg.size:
                raise ValueError(f"ImageOverlay(Light-Tool): Images must have the same size. "
                                 f"{image.size}and{bg.size} is not match")

            image = functional.to_tensor(image)
            bg = functional.to_tensor(bg)
            bg = functional.resize(bg, image.shape[-2:])
            alpha = functional.to_tensor(alpha)
            new_image = image * alpha + bg * (1 - alpha)
            new_image = new_image.squeeze(0).permute(1, 2, 0).numpy()
            image_list.append(pil2tensor(np2pil(new_image)))
        images = torch.cat(image_list, dim=0)
        return (images,)


class SimpleImageOverlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "center": ("BOOLEAN", {"default": True}),
                "left": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "top": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine_images"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Overlay one image on top of another to create a composite image"

    @staticmethod
    def combine_images(foreground: torch.Tensor, background: torch.Tensor, center, left, top):

        foreground_pil = tensor2pil(foreground)
        if foreground_pil.mode != "RGBA":
            foreground_pil = foreground_pil.convert("RGBA")
        background_pil = tensor2pil(background)

        if center:
            bg_width, bg_height = background_pil.size
            fg_width, fg_height = foreground_pil.size
            left = (bg_width - fg_width) // 2
            top = (bg_height - fg_height) // 2

        background_pil.paste(foreground_pil, (left, top), foreground_pil)
        result = pil2tensor(background_pil)
        return (result, )


class AddBackgroundV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "use_hex": ("BOOLEAN", {"default": True}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "square": ("BOOLEAN", {"default": False}),
                "left_margin": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "right_margin": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "top_margin": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "bottom_margin": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_background_v2"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Add solid color background to transparent image"

    @staticmethod
    def add_background_v2(image, color_hex, use_hex, R, G, B, square, left_margin, right_margin, top_margin,
                          bottom_margin):

        if use_hex and not (color_hex.startswith("#") and len(color_hex) == 7):
            raise ValueError("AddBackgroundV2(Light-Tool): Invalid hexadecimal color value")

        image_list = []
        for img in image:
            overlay = tensor2pil(img)
            width, height = overlay.size
            if square:
                if width > height:
                    top_margin += (width - height) // 2
                    bottom_margin += (width - height) // 2
                else:
                    left_margin += (height - width) // 2
                    right_margin += (height - width) // 2

            background_width = width + left_margin + right_margin
            background_height = height + top_margin + bottom_margin
            if use_hex:
                rgb_background_color = hex_to_rgb(color_hex)
            else:
                rgb_background_color = (R, G, B)

            background = Image.new('RGB', (background_width, background_height), rgb_background_color)
            background.paste(overlay, (left_margin, top_margin), overlay)
            new_image = pil2tensor(background)
            image_list.append(new_image)

        new_image = torch.cat(image_list, dim=0)
        return (new_image,)


class IsTransparent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.99, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "is_transparent"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageInfo'
    DESCRIPTION = "Detect if an image is transparent"

    @staticmethod
    def is_transparent(image: torch.Tensor, threshold: float):
        img = tensor2pil(image)
        if img.mode == 'RGBA':
            img_array = np.array(img)
            alpha_channel = img_array[:, :, 3] / 255.0
            transparency_ratio = np.mean(alpha_channel < 1)
            return (transparency_ratio > threshold,)
        elif img.mode == 'P':
            try:
                transparent_index = img.info['transparency']
                color = img.getcolor(transparent_index)
                return (color[3] == 0,)
            except KeyError:
                return (False,)
        else:
            return (False,)


class PhantomTankEffect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "offset": ("INT", {"default": 128, "min": 0, "max": 255, "display": "slider"}),
                "alpha_min": ("INT", {"default": 1, "min": 0, "max": 255, "display": "slider"}),
                "alpha_max": ("INT", {"default": 255, "min": 0, "max": 255, "display": "slider"}),
                "v_min": ("INT", {"default": 0, "min": 0, "max": 255, "display": "slider"}),
                "v_max": ("INT", {"default": 255, "min": 0, "max": 255, "display": "slider"}),
                "gray_method": (["luminosity", "average", "max", "min", "custom"], {"default": "luminosity"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_images"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Creates a 'Phantom Tank' effect image"

    @staticmethod
    def process_images(image1: torch.Tensor, image2: torch.Tensor, offset: int, alpha_min: int, alpha_max: int,
                       v_min: int, v_max: int, gray_method: int):

        img1 = tensor2pil(image1)
        img2 = tensor2pil(image2)

        width = max(img1.width, img2.width)
        height = max(img1.height, img2.height)

        img1_gray = to_gray(img1, method=gray_method)
        img2_gray = to_gray(img2, method=gray_method)

        result_img = Image.new("RGBA", (width, height))

        pixels1: Any = img1_gray.load()
        pixels2: Any = img2_gray.load()
        pixels_result: Any = result_img.load()

        for y in range(height):
            for x in range(width):
                v1 = pixels1[x, y] / 2
                v2 = pixels2[x, y] / 2 + offset
                v1 = max(alpha_min, v1)
                v2 = min(alpha_max - 1, v2)
                a = alpha_max - v2 + v1
                a = min(alpha_max, max(alpha_min, a))
                v = v1 * v_max / a
                v = min(v_max, max(v_min, v))
                pixels_result[x, y] = (int(v), int(v), int(v), int(a))

        result_img = pil2tensor(result_img)
        return (result_img,)


class MaskContourExtractor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_hex": ("STRING", {"default": "#FF0000", "multiline": False}),
                "use_hex": ("BOOLEAN", {"default": True}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "contour_extractor"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageInfo'
    DESCRIPTION = "Extract contour points of the mask in the image"

    @staticmethod
    def contour_extractor(image, color_hex, use_hex, R, G, B):

        image_pil = tensor2pil(image).convert('L')
        image = np.array(image_pil).astype(np.uint8)

        if use_hex:
            rgb_background_color = hex_to_rgb(color_hex)
        else:
            rgb_background_color = (R, G, B)

        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_points = []
        for contour in contours:
            for point in contour:
                contour_points.append(list(point[0]))

        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_image, contours, -1, rgb_background_color, 1)
        result_img = np2tensor(contour_image)
        return (result_img,)


class AdvancedSolidColorBackground:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 0, "display": "number"}),
                "height": ("INT", {"default": 1024, "min": 0, "display": "number"}),
                "color_hex": ("STRING", {"default": "#FF0000", "multiline": False}),
                "use_hex": ("BOOLEAN", {"default": True}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
                "alpha": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_bg"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Generate solid color background"

    @staticmethod
    def generate_bg(color_hex, use_hex, width, height, R, G, B, mode, alpha):

        if use_hex:
            rgb_background_color = hex_to_rgb(color_hex)
        else:
            rgb_background_color = (R, G, B)

        if 'A' in mode:
            alpha_channel = alpha
            rgb_background_color += (alpha_channel,)

        image = Image.new(mode, (width, height), rgb_background_color)
        result_img = pil2tensor(image)
        return (result_img,)


class ResizeImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 0, "display": "number"}),
                "resize_method": (["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"], {"default": "LANCZOS"}),
                "mode": (["RGB", "RGBA", "L"], {"default": "RGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_img"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageCrop'
    DESCRIPTION = "Crop an image based on its width and height"

    @staticmethod
    def resize_img(image, width, height, resize_method, mode):
        image = tensor2pil(image).convert(mode)
        if width is None and height is None:
            raise ValueError("Either new_width or new_height must be provided.")
        elif width is not None and height is None:
            height = int((width / image.width) * image.height)
        elif height is not None and width is None:
            width = int((height / image.height) * image.width)

        method = {
            "LANCZOS": Image.LANCZOS,
            "BICUBIC": Image.BICUBIC,
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR
        }

        img_resized = image.resize((width, height), method[resize_method])
        result_img = pil2tensor(img_resized)
        return (result_img,)


class RGB2RGBA:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rgb2rgba"
    CATEGORY = 'ComfyUI-Light-Tool/image/transform'
    DESCRIPTION = "Convert an RGB image to RGBA format"

    @staticmethod
    def rgb2rgba(image):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert("RGBA")
            result_img = pil2tensor(img)
            image_list.append(result_img)
        new_image = torch.cat(image_list, dim=0)
        return (new_image,)


class RGBA2RGB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rgba2rgb"
    CATEGORY = 'ComfyUI-Light-Tool/image/transform'
    DESCRIPTION = "Convert an RGBA image to RGB format"

    @staticmethod
    def rgba2rgb(image):
        image_list = []
        for img in image:
            img = tensor2pil(img).convert("RGB")
            result_img = pil2tensor(img)
            image_list.append(result_img)
        new_image = torch.cat(image_list, dim=0)
        return (new_image,)


class InputText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": (
                    "STRING", {"defaultInput": False, "multiline": True, "placeholder": "Please input text"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "input_text"
    CATEGORY = 'ComfyUI-Light-Tool/Text'
    DESCRIPTION = "Input string text "

    @staticmethod
    def input_text(input_text):
        input_text = '' if not input_text else input_text.strip()
        return (input_text,)


class ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (any_type, {"defaultInput": True, "multiline": True}),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show_text"
    CATEGORY = 'ComfyUI-Light-Tool/Text'
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Show output Text"

    @staticmethod
    def show_text(text):
        output = text[0] if isinstance(text[0], list) else text
        return {"ui": {"text": output}, "result": (output,)}


class PreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": (
                    "STRING",
                    {"defaultInput": True, "multiline": True, "placeholder": "Please input video url"}
                )
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ("video",)
    FUNCTION = "preview_video"
    CATEGORY = 'ComfyUI-Light-Tool/Video'
    DESCRIPTION = "Preview video from video url"

    @staticmethod
    def preview_video(video_url):
        return {"ui": {"video_url": [video_url]}}


class LoadVideo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"defaultInput": True}),
                "server": ("STRING", {"defaultInput": False, "default": '', "placeholder": "local server address"}),
                "save_dir": ("STRING", {"defaultInput": False, "default": ''}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "video_url")
    FUNCTION = "load_video"
    CATEGORY = 'ComfyUI-Light-Tool/Video'
    DESCRIPTION = "Load the video from video url and save to your directory"

    def load_video(self, video_url, server, save_dir):
        filename = str(uuid.uuid4())
        save_file_path = os.path.join(self.output_dir, save_dir)
        os.makedirs(save_file_path, exist_ok=True)
        download_file(video_url, os.path.join(save_file_path, filename + ".mp4"))
        t13 = int(time.time() * 1000)
        server_url = server.split('?')[0].strip().rstrip('/')
        video_url = f"{server_url}/api/view?filename={filename}.mp4&type=output&subfolder={save_dir}&t={t13}"
        return os.path.join(save_dir, filename+'.mp4'), video_url


class SaveVideo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"defaultInput": True}),
                "server": ("STRING", {"defaultInput": False, "default": '', "placeholder": "local server address"}),
                "save_dir": ("STRING", {"defaultInput": False, "default": ''}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_video"
    CATEGORY = 'ComfyUI-Light-Tool/Video'
    DESCRIPTION = "Saves the video by video url to your directory"

    def save_video(self, video_url, server, save_dir):
        filename = str(uuid.uuid4())
        save_file_path = os.path.join(self.output_dir, save_dir)
        os.makedirs(save_file_path, exist_ok=True)
        download_file(video_url, os.path.join(save_file_path, filename + ".mp4"))
        t13 = int(time.time() * 1000)
        server_url = server.split('?')[0].strip().rstrip('/')
        video_url = f"{server_url}/api/view?filename={filename}.mp4&type=output&subfolder={save_dir}&t={t13}"
        return {"ui": {"video_url": [video_url]}}


class SaveVideoToAliyunOss:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.input_dir = folder_paths.get_input_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"defaultInput": True}),
                "save_name": ("STRING", {"default": ""}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "oss_access_key_id": ("STRING", {"default": ""}),
                "oss_access_key_secret": ("STRING", {"default": ""}),
                "oss_session_token": ("STRING", {"default": ""}),
                "visit_endpoint": ("STRING", {"default": ""}),
                "use_cname": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sign": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "timeout": ("INT", {"default": 0}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "save_video"
    CATEGORY = 'ComfyUI-Light-Tool/Video'
    DESCRIPTION = "Saves the video to aliyun OSS"

    def save_video(self, file, save_name, endpoint, bucket, oss_access_key_id, oss_access_key_secret,
                   oss_session_token, use_cname, visit_endpoint, sign, timeout):

        if 'http' in file:
            filename = save_name or file.split('?')[0].rsplit('/')[-1]
            file_path = download_file(file, filename)
        elif '/' not in file:
            if os.path.exists(os.path.join(self.output_dir, file)):
                file_path = os.path.join(self.output_dir, file)
            else:
                file_path = os.path.join(self.input_dir, file)
        else:
            file_path = file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'(ComfyUI-Light-Tool) {file_path} Upload file not exists')

        video_url = oss_upload(
            file_path,
            filename=os.path.basename(file_path),
            endpoint=endpoint,
            bucket=bucket,
            oss_access_key_id=oss_access_key_id,
            oss_access_key_secret=oss_access_key_secret,
            oss_session_token=oss_session_token,
            is_cname=use_cname,
            visit_endpoint=visit_endpoint,
            sign=sign,
            timeout=timeout
        )
        return (video_url,)


class SaveToAliyunOSS:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.input_dir = folder_paths.get_input_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"defaultInput": True}),
                "save_name": ("STRING", {"default": ""}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "oss_access_key_id": ("STRING", {"default": ""}),
                "oss_access_key_secret": ("STRING", {"default": ""}),
                "oss_session_token": ("STRING", {"default": ""}),
                "visit_endpoint": ("STRING", {"default": ""}),
                "use_cname": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sign": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "timeout": ("INT", {"default": 0}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_url",)
    FUNCTION = "save"
    CATEGORY = 'ComfyUI-Light-Tool/OSS'
    DESCRIPTION = "Saves the file to aliyun OSS"

    def save(self, file, save_name, endpoint, bucket, oss_access_key_id, oss_access_key_secret,
             oss_session_token, use_cname, visit_endpoint, sign, timeout):

        if 'http' in file:
            filename = save_name or file.split('?')[0].rsplit('/')[-1]
            file_path = download_file(file, filename)
        elif '/' not in file:
            if os.path.exists(os.path.join(self.output_dir, file)):
                file_path = os.path.join(self.output_dir, file)
            else:
                file_path = os.path.join(self.input_dir, file)
        else:
            file_path = file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'(ComfyUI-Light-Tool/OSS) {file_path} Upload file not exists')

        file_url = oss_upload(
            file_path,
            filename=os.path.basename(file_path),
            endpoint=endpoint,
            bucket=bucket,
            oss_access_key_id=oss_access_key_id,
            oss_access_key_secret=oss_access_key_secret,
            oss_session_token=oss_session_token,
            is_cname=use_cname,
            visit_endpoint=visit_endpoint,
            sign=sign,
            timeout=timeout
        )
        return (file_url,)


class GetImageSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "output_size": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "size")
    FUNCTION = "image_size"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageInfo'
    DESCRIPTION = "Get image base info"

    @staticmethod
    def image_size(image, output_size):
        image = tensor2pil(image)
        file_size = 0
        if output_size:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                temp_file_path = tmp.name + '.png'
                image.save(temp_file_path)
                file_size = os.path.getsize(temp_file_path)
                print(f"The size of the image file is: {file_size} bytes")
        return image.width, image.height, file_size


class ImageConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"})
            },
            "optional": {
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_concat"
    CATEGORY = 'ComfyUI-Light-Tool/image/compositing'
    DESCRIPTION = "Concatenates the N input images into a 1xN or Nx1 grid"

    @staticmethod
    def image_concat(direction, **kwargs):
        images = [tensor2pil(x) for x in kwargs.values()]

        if direction == 'horizontal':
            height = images[0].height
            for img in images:
                if img.height != height:
                    raise ValueError(
                        f"(Light-Tool/ImageConcat) All images must have the same height. {img.height} and {height}"
                    )
            total_width = sum(img.width for img in images)
            new_image = Image.new('RGB', (total_width, height))
        elif direction == 'vertical':
            width = images[0].width
            for img in images:
                if img.width != width:
                    raise ValueError(
                        f"(Light-Tool/ImageConcat) All images must have the same width. {img.width} and {width}"
                    )
            total_height = sum(img.height for img in images)
            new_image = Image.new('RGB', (width, total_height))
        else:
            raise ValueError("(Light-Tool/ImageConcat) Direction must be 'horizontal' or 'vertical'.")

        x_offset = 0
        y_offset = 0
        for img in images:
            if direction == 'horizontal':
                new_image.paste(img, (x_offset, 0))
                x_offset += img.width
            elif direction == 'vertical':
                new_image.paste(img, (0, y_offset))
                y_offset += img.height

        result_img = pil2tensor(new_image)
        mask = np2tensor(np.array(new_image)[:, :, 0])
        return result_img, mask


class TextConnect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "string_list": (any_type, {"default": ""}),
                "string_1": ("STRING", ),
                "string_2": ("STRING", ),
                "string_3": ("STRING", ),
                "string_4": ("STRING", ),
                "delimiter": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_connect"
    CATEGORY = 'ComfyUI-Light-Tool/Text'
    DESCRIPTION = "Connect multiple text strings"

    @staticmethod
    def text_connect(delimiter, **kwargs):
        string_list1 = kwargs.get('string_list')
        string_list2 = [x for x in kwargs.values() if x and not isinstance(x, list)]
        all_string_list = string_list1 + string_list2 if string_list1 else string_list2
        return (delimiter.join(all_string_list),)


class SimpleTextConnect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": (any_type, {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ""}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_connect"
    CATEGORY = 'ComfyUI-Light-Tool/Text'
    DESCRIPTION = "Connect multiple text strings"

    @staticmethod
    def text_connect(string1, string2, delimiter):
        if type(string1) == list:
            connect_result = []
            for str1 in string1:
                result = str1 + delimiter + string2
                connect_result.append(result)
        else:
            connect_result = string1 + delimiter + string2
        return (connect_result,)


class InputTextList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "string_1": ("STRING", ),
                "string_2": ("STRING", ),
                "string_3": ("STRING", ),
                "string_4": ("STRING", ),
            }
        }

    OUTPUT_IS_LIST = (True, )
    OUTPUT_NODE = True
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("strings",)
    FUNCTION = "input_text_list"
    CATEGORY = 'ComfyUI-Light-Tool/Text'
    DESCRIPTION = "Connect multiple text strings"

    @staticmethod
    def input_text_list(**kwargs):
        return ([[x for x in kwargs.values() if x]],)


class MorphologicalTF:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "option": (["dilate", "erode"], {"default": "dilate"}),
                "kernel_x": ("INT", {"default": 3, "min": 0, "display": "number"}),
                "kernel_y": ("INT", {"default": 3, "min": 0, "display": "number"}),
                "iterations": ("INT", {"default": 1, "min": 0, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "morphology_process"
    CATEGORY = 'ComfyUI-Light-Tool/image/transform'
    DESCRIPTION = "Perform morphological transformations on images"

    @staticmethod
    def morphology_process(image, option, kernel_x, kernel_y, iterations):

        image_pil = tensor2pil(image).convert('L')
        image = np.array(image_pil).astype(np.uint8)
        if option == "dilate":
            result_image = dilate_image(image, (kernel_x, kernel_y), iterations)
        else:
            result_image = erode_image(image, (kernel_x, kernel_y), iterations)
        result_img = np2tensor(result_image)
        return (result_img,)


class Hex2Rgb:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("R", "G", "B")
    FUNCTION = "hex2rgb"
    CATEGORY = 'ComfyUI-Light-Tool/image/ImageInfo'
    DESCRIPTION = "Hex color code convert to RGB code"

    @staticmethod
    def hex2rgb(color_hex):
        if "#000000" == color_hex:
            return 0, 0, 0
        elif "#ffffff".lower() == color_hex:
            return 255, 255, 255
        else:
            return hex_to_rgb(color_hex)


class ScaleImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 0, "display": "number"}),
                "height": ("INT", {"default": 1024, "min": 0, "display": "number"}),
                "mode": (["AUTO", "STRETCH", "FILL", "PAD"], {"default": "AUTO"}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "A": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "align": (["center", "top-left", "top-right", "bottom-left", "bottom-right"], {"default": "center"}),
                "resample": (["LANCZOS", "NEAREST", "BILINEAR", "BICUBIC", "BOX", "HAMMING"], {"default": "LANCZOS"}),
                "aspect_tolerance": ("FLOAT", {"default": 0.01, "min": 0, "display": "number"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale_image"
    CATEGORY = 'ComfyUI-Light-Tool/image/transform'
    DESCRIPTION = ""

    @staticmethod
    def scale_image(
            image, width, height, mode, R, G, B, A, align, resample, aspect_tolerance):

        image_pil = tensor2pil(image)
        sale_mode_dict = {
            "AUTO": ScaleMode.AUTO,
            "STRETCH": ScaleMode.STRETCH,
            "FILL": ScaleMode.FILL,
            "PAD": ScaleMode.PAD
        }

        resample_dict = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "NEAREST": Image.Resampling.NEAREST,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
            "BOX": Image.Resampling.BOX,
            "HAMMING": Image.Resampling.HAMMING
        }

        scale_img = scale_image(
            image_pil,
            (width, height),
            background=(R, G, B, A),
            mode=sale_mode_dict[mode],
            align=align,
            resample=resample_dict[resample],
            aspect_tolerance=aspect_tolerance
        )
        result_img = pil2tensor(scale_img)
        return (result_img,)


class UpscaleImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 0, "display": "number"}),
                "height": ("INT", {"default": 1024, "min": 0, "display": "number"}),
                "mode": (["AUTO", "STRETCH", "FILL", "PAD"], {"default": "AUTO"}),
                "R": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "A": ("INT", {"default": 255, "min": 0, "max": 255, "display": "number"}),
                "align": (["center", "top-left", "top-right", "bottom-left", "bottom-right"], {"default": "center"}),
                "resample": (["LANCZOS", "NEAREST", "BILINEAR", "BICUBIC", "BOX", "HAMMING"], {"default": "LANCZOS"}),
                "aspect_tolerance": ("FLOAT", {"default": 0.01, "min": 0, "display": "number"}),
                "sharpen": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_image"
    CATEGORY = 'ComfyUI-Light-Tool/image/transform'
    DESCRIPTION = ""

    @staticmethod
    def upscale_image(
            image, width, height, mode, R, G, B, A, align, resample, aspect_tolerance, sharpen):

        image_pil = tensor2pil(image)
        upsale_mode_dict = {
            "AUTO": UpscaleMode.AUTO,
            "STRETCH": UpscaleMode.STRETCH,
            "FILL": UpscaleMode.FILL,
            "PAD": UpscaleMode.PAD
        }

        resample_dict = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "NEAREST": Image.Resampling.NEAREST,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
            "BOX": Image.Resampling.BOX,
            "HAMMING": Image.Resampling.HAMMING
        }

        upscale_img = upscale_image(
            image_pil,
            (width, height),
            background=(R, G, B, A),
            mode=upsale_mode_dict[mode],
            align=align,
            resample=resample_dict[resample],
            aspect_tolerance=aspect_tolerance,
            sharpen=sharpen
        )
        result_img = pil2tensor(upscale_img)
        return (result_img,)


NODE_CLASS_MAPPINGS = {
    "Light-Tool: InputText": InputText,
    "Light-Tool: InputTextList": InputTextList,
    "Light-Tool: ShowText": ShowText,
    "Light-Tool: TextConnect": TextConnect,
    "Light-Tool: SimpleTextConnect": SimpleTextConnect,
    "Light-Tool: LoadImage": LoadImage,
    "Light-Tool: LoadImageFromURL": LoadImageFromURL,
    "Light-Tool: LoadImagesFromDir": LoadImagesFromDir,
    "Light-Tool: GetImageSize": GetImageSize,
    "Light-Tool: Hex2Rgb": Hex2Rgb,
    "Light-Tool: MaskToImage": MaskToImage,
    "Light-Tool: ImageToMask": ImageToMask,
    "Light-Tool: InvertMask": InvertMask,
    "Light-Tool: RGB2RGBA": RGB2RGBA,
    "Light-Tool: RGBA2RGB": RGBA2RGB,
    "Light-Tool: MorphologicalTF": MorphologicalTF,
    "Light-Tool: ImageMaskApply": ImageMaskApply,
    "Light-Tool: SimpleImageOverlay": SimpleImageOverlay,
    "Light-Tool: ImageOverlay": ImageOverlay,
    "Light-Tool: BoundingBoxCropping": BoundingBoxCropping,
    "Light-Tool: AddBackground": AddBackground,
    "Light-Tool: AddBackgroundV2": AddBackgroundV2,
    "Light-Tool: ResizeImage": ResizeImage,
    "Light-Tool: UpscaleImage": UpscaleImage,
    "Light-Tool: ScaleImage": ScaleImage,
    "Light-Tool: IsTransparent": IsTransparent,
    "Light-Tool: MaskBoundingBoxCropping": MaskBoundingBoxCropping,
    "Light-Tool: MaskImageToTransparent": MaskImageToTransparent,
    "Light-Tool: PhantomTankEffect": PhantomTankEffect,
    "Light-Tool: MaskContourExtractor": MaskContourExtractor,
    "Light-Tool: SolidColorBackground": AdvancedSolidColorBackground,
    "Light-Tool: ImageConcat": ImageConcat,
    "Light-Tool: PreviewVideo": PreviewVideo,
    "Light-Tool: LoadVideo": LoadVideo,
    "Light-Tool: SaveVideo": SaveVideo,
    "Light-Tool: SaveToAliyunOSS": SaveToAliyunOSS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: InputText": "Light-Tool: Input Text",
    "Light-Tool: ShowText": "Light-Tool: Show Text",
    "Light-Tool: InputTextList": "Light-Tool: Input Text List",
    "Light-Tool: TextConnect": "Light-Tool: Connect Text Strings",
    "Light-Tool: SimpleTextConnect": "Light-Tool: Simple Connect Text Strings",
    "Light-Tool: LoadImage": "Light-Tool: Load Image",
    "Light-Tool: LoadImageFromURL": "Light-Tool: Load Image From URL",
    "Light-Tool: LoadImagesFromDir": "Light-Tool: Load Image List",
    "Light-Tool: GetImageSize": "Light-Tool: Get Image Size",
    "Light-Tool: Hex2Rgb": "Light-Tool: Hex to Rgb",
    "Light-Tool: MaskToImage": "Light-Tool: Mask to Image",
    "Light-Tool: ImageToMask": "Light-Tool: Image to Mask",
    "Light-Tool: InvertMask": "Light-Tool: Invert Mask",
    "Light-Tool: RGB2RGBA": "Light-Tool: RGB To RGBA",
    "Light-Tool: RGBA2RGB": "Light-Tool: RGBa To RGB",
    "Light-Tool: MorphologicalTF": "Light-Tool: Morphological Transform",
    "Light-Tool: ImageMaskApply": "Light-Tool: Extract Transparent Image",
    "Light-Tool: SimpleImageOverlay": "Light-Tool: Simple Image Overlay",
    "Light-Tool: ImageOverlay": "Light-Tool: Image Overlay",
    "Light-Tool: BoundingBoxCropping": "Light-Tool: Bounding Box Cropping",
    "Light-Tool: AddBackground": "Light-Tool: Add solid color background",
    "Light-Tool: AddBackgroundV2": "Light-Tool: Add solid color background V2",
    "Light-Tool: ResizeImage": "Light-Tool: Resize Image",
    "Light-Tool: UpscaleImage": "Light-Tool: Upscale Image",
    "Light-Tool: ScaleImage": "Light-Tool: Scale Image",
    "Light-Tool: IsTransparent": "Light-Tool: Is Transparent",
    "Light-Tool: MaskBoundingBoxCropping": "Light-Tool: Mask Bounding Box Cropping",
    "Light-Tool: MaskImageToTransparent": "Light-Tool: Mask Background to Transparent",
    "Light-Tool: PhantomTankEffect": "Light-Tool: Generate PhantomTankEffect",
    "Light-Tool: SolidColorBackground": "Light-Tool: SolidColorBackground",
    "Light-Tool: ImageConcat": "Light-Tool: Image Concat",
    "Light-Tool: PreviewVideo": "Light-Tool: Preview Video",
    "Light-Tool: LoadVideo": "Light-Tool: Load Video",
    "Light-Tool: SaveVideo": "Light-Tool: Save Video",
    "Light-Tool: SaveToAliyunOSS": "Light-Tool: Save File To Aliyun OSS"
}