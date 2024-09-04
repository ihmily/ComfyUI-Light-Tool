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
import httpx
from PIL import ImageSequence, ImageOps

from torchvision.transforms import functional
import folder_paths
import node_helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from light_tool_utils import *


class LoadImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
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
    CATEGORY = 'ComfyUI-Light-Tool'

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
                i = i.point(lambda i: i * (1 / 255))

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
            image = torch.from_numpy(image)[None,]
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
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
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
    CATEGORY = "ComfyUI-Light-Tool"

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


class ImageMaskApply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = 'ComfyUI-Light-Tool'

    @staticmethod
    def run(image, mask):
        image_list = []
        for _image, _mask in zip(image, mask):
            image_pil = tensor2pil(_image).convert('RGB')
            mask_pil = tensor2pil(_mask).convert('L')
            image_size = image_pil.size
            mask_size = mask_pil.size
            if mask_pil.size != image_pil.size:
                raise ValueError(f"ImageMaskApply(Light-Tool): Images must have the same size. "
                                 f"{image_size}and{mask_size} is not match")
            image_pil = RGB2RGBA(image_pil, mask_pil)
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
    CATEGORY = 'ComfyUI-Light-Tool'

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
                "channel": (["red", "green", "blue", "alpha"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_to_mask"
    CATEGORY = 'ComfyUI-Light-Tool'

    @staticmethod
    def image_to_mask(image, channel):
        mask_list = []
        for img in image:
            channels = ["red", "green", "blue", "alpha"]
            mask = img[:, :, :, channels.index(channel)]
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
    CATEGORY = 'ComfyUI-Light-Tool'

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
    CATEGORY = 'ComfyUI-Light-Tool'

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
    CATEGORY = 'ComfyUI-Light-Tool'

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
    CATEGORY = 'ComfyUI-Light-Tool'

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
    CATEGORY = 'ComfyUI-Light-Tool'

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
    CATEGORY = 'ComfyUI-Light-Tool'

    @staticmethod
    def combine_images(origin_image: torch.Tensor, overlay_image: torch.Tensor, overlay_mask: torch.Tensor):
        image_list = []
        for img in origin_image:
            image = tensor2pil(overlay_image).convert('RGB')
            bg = tensor2pil(img).convert('RGB')
            alpha = tensor2pil(overlay_mask).convert('L')

            image = functional.to_tensor(image)
            bg = functional.to_tensor(bg)
            bg = functional.resize(bg, image.shape[-2:])
            alpha = functional.to_tensor(alpha)
            new_image = image * alpha + bg * (1 - alpha)
            new_image = new_image.squeeze(0).permute(1, 2, 0).numpy()
            image_list.append(pil2tensor(np2pil(new_image)))
        images = torch.cat(image_list, dim=0)
        return (images,)


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
    CATEGORY = 'ComfyUI-Light-Tool'

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
    CATEGORY = 'ComfyUI-Light-Tool'

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


NODE_CLASS_MAPPINGS = {
    "Light-Tool: LoadImage": LoadImage,
    "Light-Tool: LoadImageFromURL": LoadImageFromURL,
    "Light-Tool: MaskToImage": MaskToImage,
    "Light-Tool: ImageToMask": ImageToMask,
    "Light-Tool: InvertMask": InvertMask,
    "Light-Tool: ImageMaskApply": ImageMaskApply,
    "Light-Tool: ImageOverlay": ImageOverlay,
    "Light-Tool: BoundingBoxCropping": BoundingBoxCropping,
    "Light-Tool: AddBackground": AddBackground,
    "Light-Tool: AddBackgroundV2": AddBackgroundV2,
    "Light-Tool: IsTransparent": IsTransparent,
    "Light-Tool: MaskBoundingBoxCropping": MaskBoundingBoxCropping,
    "Light-Tool: MaskImageToTransparent": MaskImageToTransparent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: LoadImage": "Light-Tool: Load Image",
    "Light-Tool: LoadImageFromURL": "Light-Tool: Load Image From URL",
    "Light-Tool: MaskToImage": "Light-Tool: Mask to Image",
    "Light-Tool: ImageToMask": "Light-Tool: Image to Mask",
    "Light-Tool: InvertMask": "Light-Tool: Invert Mask",
    "Light-Tool: ImageMaskApply": "Light-Tool: Extract Transparent Image",
    "Light-Tool: ImageOverlay": "Light-Tool: Image Overlay",
    "Light-Tool: BoundingBoxCropping": "Light-Tool: Bounding Box Cropping",
    "Light-Tool: AddBackground": "Light-Tool: Add solid color background",
    "Light-Tool: AddBackgroundV2": "Light-Tool: Add solid color background V2",
    "Light-Tool: IsTransparent": "Light-Tool: Is Transparent",
    "Light-Tool: MaskBoundingBoxCropping": "Light-Tool: Mask Bounding Box Cropping",
    "Light-Tool: MaskImageToTransparent": "Light-Tool: Mask Background to Transparent",
}
