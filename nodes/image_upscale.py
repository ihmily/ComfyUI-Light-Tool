from PIL import Image, ImageOps, ImageFilter
from enum import Enum


class UpscaleMode(Enum):
    """
    Enum for upscale modes.
    STRETCH = Stretch to target size forcibly.
    FILL    = Keep aspect ratio and fill the target area with cropping.
    PAD     = Keep aspect ratio and add a background padding.
    AUTO    = Automatically choose the best mode based on aspect ratios.
    """
    STRETCH = 1
    FILL = 2
    PAD = 3
    AUTO = 4


def upscale_image(
        input_image: str | Image.Image,
        target_size: tuple[int, int],
        mode: UpscaleMode = UpscaleMode.AUTO,
        background: tuple[int, int, int, int] = (0, 0, 0, 0),
        align: str = 'center',
        resample: Image.Resampling = Image.Resampling.LANCZOS,
        aspect_tolerance: float = 0.01,
        sharpen: bool = True
) -> Image.Image:
    """
    Advanced image upscaling function.

    Parameters:
    - sharpen: Whether to apply a sharpen filter (default is True).
    """
    if isinstance(input_image, str):
        img = Image.open(input_image).convert("RGBA")
    else:
        img = input_image.convert("RGBA")

    orig_w, orig_h = img.size
    target_w, target_h = target_size

    # Logic for auto mode
    if mode == UpscaleMode.AUTO:
        orig_ratio = orig_w / orig_h
        target_ratio = target_w / target_h

        if abs(orig_ratio - target_ratio) <= aspect_tolerance:
            mode = UpscaleMode.STRETCH
        else:
            mode = UpscaleMode.PAD

    # Handling different modes
    if mode == UpscaleMode.STRETCH:
        img = img.resize(target_size, resample=resample)
    elif mode == UpscaleMode.FILL:
        # Calculate maximum scaling ratio
        ratio = max(target_w / orig_w, target_h / orig_h)
        scaled_size = (int(orig_w * ratio), int(orig_h * ratio))

        img = img.resize(scaled_size, resample=resample)
        img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
    elif mode == UpscaleMode.PAD:
        ratio = min(target_w / orig_w, target_h / orig_h)
        scaled_size = (int(orig_w * ratio), int(orig_h * ratio))

        img = img.resize(scaled_size, resample=resample)
        canvas = Image.new("RGBA", target_size, background)
        offset = calculate_alignment_offset(scaled_size, target_size, align)
        canvas.paste(img, offset, mask=img)
        img = canvas

    # Sharpening process
    if sharpen and (target_w > orig_w or target_h > orig_h):
        img = img.filter(ImageFilter.SHARPEN)

    return img


def calculate_alignment_offset(src_size: tuple[int, int], dst_size: tuple[int, int], align: str) -> tuple[int, int]:
    """
    Calculate alignment offsets.
    """
    src_w, src_h = src_size
    dst_w, dst_h = dst_size

    if 'left' in align:
        x = 0
    elif 'right' in align:
        x = dst_w - src_w
    else:
        x = (dst_w - src_w) // 2

    if 'top' in align:
        y = 0
    elif 'bottom' in align:
        y = dst_h - src_h
    else:
        y = (dst_h - src_h) // 2

    return x, y
