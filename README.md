# ComfyUI-Light-Tool

**English** / [简体中文](./README_ZH.md)

A suite of lightweight and practical custom tool nodes for ComfyUI to enhance workflow efficiency and flexibility.

---

## 📦 Installation Guide

### 1. Manual Installation

```bash
# Navigate to the custom_nodes directory of ComfyUI
cd ComfyUI/custom_nodes

# Clone this repository
git clone https://github.com/ihmily/ComfyUI-Light-Tool.git
pip install -r requirements.txt

# Restart ComfyUI
```

### 2. Installation via ComfyUI-Manager

Automatic installation through ComfyUI-Manager:

1. Go to the **Manager** panel in the ComfyUI interface.
2. Under the **Custom Nodes** tab, search for `ComfyUI-Light-Tool`.
3. Click **Install** and restart ComfyUI.

---

## 🎨 Functional Categories and Node List

### 1. Image Processing

#### 1.1 Load Images

| Node Name               | Function Description                                                              |
| ----------------------- | --------------------------------------------------------------------------------- |
| **Load Image**          | Load local image files (supports preserving RGBA transparency).                   |
| **Load Image From URL** | Download and load images from a URL (supports comma-separated multiple URLs).     |
| **Load Image List**     | Batch load images from a directory (supports start index and count configuration).|

#### 1.2 Image Compositing & Overlaying

| Node Name                          | Function Description                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| **Image Overlay**                  | Overlay one image onto another (requires mask, image sizes must match).           |
| **Simple Image Overlay**           | Freely overlay images (supports centering or custom offset positioning).          |
| **Image Concat**                   | Concatenate up to 5 images horizontally or vertically.                            |
| **Add Solid Color Background**     | Add a solid color background to transparent images (supports HEX or RGB color).   |
| **Add Solid Color Background V2**  | Add a solid color background with configurable margins and auto-square mode.      |
| **Solid Color Background**         | Generate a custom solid color background image (supports RGB / RGBA / grayscale L).|

#### 1.3 Image Scaling & Resizing

| Node Name                     | Function Description                                                              |
| ----------------------------- | --------------------------------------------------------------------------------- |
| **Resize Image**              | Resize image to specified width and height directly.                              |
| **Resize Image V2**           | Proportionally resize image by a fixed width or height.                           |
| **Resize Image By Ratio**     | Scale image by a ratio coefficient.                                               |
| **Resize Image By Max Size**  | Proportionally shrink image to ensure dimensions don't exceed the specified max.  |
| **Resize Image By Min Size**  | Proportionally enlarge image to ensure dimensions don't fall below the specified min. |
| **Scale Image**               | Scale image with multiple modes (AUTO / STRETCH / FILL / PAD).                    |
| **Upscale Image**             | Upscale image with multiple modes, supports sharpening (only applies to upscaling).|

#### 1.4 Image Cropping

| Node Name                  | Function Description                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------- |
| **Crop Image**             | Crop image by left / top / right / bottom margins.                                      |
| **Safe Image Crop**        | Crop image from center to the nearest specified multiple (e.g. 64) for VAE compatibility.|
| **Bounding Box Cropping**  | Auto-crop based on the minimum bounding rectangle of transparent objects in the image.  |

#### 1.5 Mask Operations

| Node Name                           | Function Description                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| **Image to Mask**                   | Convert a specified color channel (R/G/B/Alpha) of an image into a mask.         |
| **Mask to Image**                   | Convert a mask into a grayscale image.                                            |
| **Invert Mask**                     | Invert the white and black regions of a mask.                                     |
| **Extract Transparent Image**       | Use a mask to extract the foreground, outputting an image with alpha channel.     |
| **Mask Background to Transparent**  | Convert non-masked regions (non-white areas) in a mask image to transparency.    |
| **Mask Bounding Box Cropping**      | Crop a mask image based on the bounding rectangle of the mask's valid region.    |
| **Morphological Transform**         | Perform morphological operations on masks (dilate / erode) with configurable kernel size and iterations. |
| **Mask Contour Extractor**          | Extract mask contours and draw them on the image in a specified color.            |

#### 1.6 Color Channel Conversion

| Node Name       | Function Description                        |
| --------------- | ------------------------------------------- |
| **RGB to RGBA** | Convert an RGB image to RGBA format.        |
| **RGBA to RGB** | Convert an RGBA image to RGB format.        |

#### 1.7 Image Information

| Node Name            | Function Description                                            |
| -------------------- | --------------------------------------------------------------- |
| **Get Image Size**   | Get the width, height, and file size (bytes) of an image.       |
| **Get Images Count** | Get the number of images in a batch.                            |
| **Get Side Length**  | Get the pixel length of the longest or shortest side of an image.|
| **Is Transparent**   | Detect whether an image contains transparent regions (configurable threshold). |

---

### 2. Text Processing

| Node Name                           | Function Description                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| **Input Text**                      | Input a single text string.                                                       |
| **Input Text List**                 | Input up to 4 text strings combined into a list output.                           |
| **Connect Text Strings**            | Merge a list and multiple strings into one string with a delimiter.               |
| **Simple Connect Text Strings**     | Simply connect two strings with a delimiter (supports batch list processing).     |
| **Text Replace**                    | Replace text content (supports plain matching and regex, case-insensitive option).|
| **Show Text**                       | Display text content in the node panel.                                           |
| **Show Anything**                   | Display any type of data in the node panel (text, numbers, lists, etc.).          |

---

### 3. Video Related

| Node Name                     | Function Description                                                              |
| ----------------------------- | --------------------------------------------------------------------------------- |
| **Load Video**                | Download a video from a URL and save it to a directory, returns path and preview URL. |
| **Preview Video**             | Input a video URL and embed a preview player in the node panel.                   |
| **Save Video**                | Save a video from a URL to a specified local directory.                           |
| **Save Video to Aliyun OSS**  | Upload a video file (local path or URL) to Aliyun OSS.                            |

---

### 4. Data Processing

| Node Name                      | Function Description                                                              |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Hex to RGB**                 | Convert a hexadecimal color code to R / G / B integer values.                     |
| **RGB to Hex**                 | Convert R / G / B integer values to a hexadecimal color code string.              |
| **Calculate**                  | Perform addition, subtraction, multiplication, division, and modulo on two numbers; outputs string / int / float simultaneously. |
| **Convert Num Type**           | Convert a value to integer / float / string type.                                 |
| **KeyValue**                   | Extract a value from a JSON string by key path (supports dot-notation and array indexing). |
| **Serialize JSON Object**      | Serialize a JSON object into a JSON string.                                       |
| **Deserialize JSON String**    | Deserialize a JSON string into a JSON object.                                     |

---

### 5. Practical Tools

| Node Name                    | Function Description                                                              |
| ---------------------------- | --------------------------------------------------------------------------------- |
| **PhantomTankEffect**        | Generate a "Phantom Tank" effect image with adjustable offset, grayscale method, etc. |
| **Save to Aliyun OSS**       | Upload files (images/videos/etc.) to Aliyun OSS, supports signed URLs.           |
| **Save Metadata**            | Write metadata (workflow / prompt etc.) into a PNG image and save it.             |
| **Load Metadata From URL**   | Download an image from a URL and read the metadata embedded in it.                |

---

## 🛠️ Development and Contribution

- **Feedback**: Submit issues via [GitHub Issues](https://github.com/ihmily/ComfyUI-Light-Tool/issues).
- **Code Contribution**: Fork the repository and submit a Pull Request.
- **Documentation Updates**: Report any inaccuracies or omissions in the descriptions.

---

## 📖 License

This project is licensed under the [MIT License](LICENSE).