from PIL import Image
from enum import Enum


class ScaleMode(Enum):
    """
    Enum for scale modes.
    FIT    = Fit within target size while keeping aspect ratio.
    FILL   = Fill the target area, cropping if necessary to maintain aspect ratio.
    STRETCH = Stretch to target size forcibly.
    PAD    = Keep aspect ratio and add padding with background color.
    AUTO   = Automatically choose the best mode based on aspect ratios.
    """
    FIT = 1
    FILL = 2
    STRETCH = 3
    PAD = 4
    AUTO = 5


def scale_image(
        input_image: str | Image.Image,
        target_size: tuple[int, int],
        mode: ScaleMode = ScaleMode.AUTO,
        background: tuple[int, int, int, int] = (0, 0, 0, 0),
        align: str = 'center',
        crop_box: tuple[int, int, int, int] | None = None,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
        aspect_tolerance: float = 0.01
) -> Image.Image:
    """
    Advanced image scaling function.

    Parameters:
    - input_image: PIL.Image object or file path.
    - target_size: Target size as a tuple (width, height).
    - mode: Scaling mode from ScaleMode enum.
    - background: Background fill color in RGBA format.
    - align: Alignment of the image ('center', 'top-left', 'top-right', etc.).
    - crop_box: Custom crop box for FILL mode (left, upper, right, lower).
    - resample: Resampling filter (default is LANCZOS).
    - aspect_tolerance: Aspect ratio tolerance (default is 0.01).

    Returns:
    - A PIL.Image object.
    """
    # If input is a string, open the image file.
    if isinstance(input_image, str):
        img = Image.open(input_image).convert("RGBA")
    else:
        img = input_image.convert("RGBA")

    orig_width, orig_height = img.size
    target_width, target_height = target_size

    # Calculate aspect ratios
    orig_ratio = orig_width / orig_height
    target_ratio = target_width / target_height

    if mode == ScaleMode.STRETCH:
        # Directly stretch to fit
        return img.resize(target_size, resample=resample)

    elif mode == ScaleMode.FIT:
        # Resize to fit within the target area while maintaining aspect ratio
        img.thumbnail(target_size, resample=resample)
        return img

    elif mode == ScaleMode.FILL:
        # Resize and crop to fill the target area
        if crop_box:
            # Use custom crop box
            img = img.crop(crop_box)
        else:
            # Auto-calculate crop box
            if orig_ratio > target_ratio:
                # Original image is wider than target; crop sides
                new_height = orig_height
                new_width = int(target_ratio * new_height)
            else:
                # Original image is taller than target; crop top/bottom
                new_width = orig_width
                new_height = int(new_width / target_ratio)

            # Calculate crop coordinates
            left = (orig_width - new_width) // 2
            top = (orig_height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            img = img.crop((left, top, right, bottom))

        return img.resize(target_size, resample=resample)

    elif mode == ScaleMode.PAD:
        # Resize while maintaining aspect ratio and pad with background
        img.thumbnail(target_size, resample=resample)

        # Create canvas with background color
        canvas = Image.new("RGBA", target_size, background)

        # Calculate paste position
        offset = calculate_alignment_offset(img.size, target_size, align)

        canvas.paste(img, offset, mask=img)
        return canvas

    elif mode == ScaleMode.AUTO:
        # Automatically determine the best mode based on aspect ratios
        ratio_diff = abs(orig_ratio - target_ratio)

        if ratio_diff <= aspect_tolerance:
            # Aspect ratios are close; use stretch mode
            return scale_image(img, target_size, ScaleMode.STRETCH, resample=resample)
        else:
            # Aspect ratios differ significantly; prefer content preservation (PAD mode)
            return scale_image(
                img, target_size,
                mode=ScaleMode.PAD,
                background=background,
                align=align,
                resample=resample
            )

    else:
        raise ValueError("Unsupported scale mode")


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
