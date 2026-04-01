# ComfyUI-Light-Tool  

简体中文 / [English](./README.md)

大量轻量实用的 ComfyUI 自定义工具节点，提升工作流效率与灵活性。

---

## 📦 安装指南  
### 1. 手动安装  
```bash
# 进入 ComfyUI 的 custom_nodes 目录
cd ComfyUI/custom_nodes

# 克隆本仓库
git clone https://github.com/ihmily/ComfyUI-Light-Tool.git
pip install -r requirements.txt

# 重启 ComfyUI
```

### 2. 使用 ComfyUI-Manager 安装  
通过 ComfyUI-Manager 自动安装：  
1. 进入 ComfyUI 界面的 **Manager** 面板。  
2. 在 **Custom Nodes** 标签下搜索 `ComfyUI-Light-Tool`。  
3. 点击 **Install** 并重启 ComfyUI。

---

# 🎨 功能分类与节点列表  

### 一、图像处理  

#### 1.1 加载图像
| 节点名称                | 功能描述                                             |
| ----------------------- | ---------------------------------------------------- |
| **Load Image**          | 加载本地图片文件（支持保持 RGBA 透明度）。           |
| **Load Image From URL** | 从 URL 下载并加载图片（支持逗号分隔多个 URL）。      |
| **Load Image List**     | 批量加载指定目录中的图片（支持设置起始索引和数量）。 |

#### 1.2 图像合成与叠加
| 节点名称                          | 功能描述                                                  |
| --------------------------------- | --------------------------------------------------------- |
| **Image Overlay**                 | 将一张图片叠加到另一张上（需蒙版，图片尺寸需一致）。      |
| **Simple Image Overlay**          | 任意图片自由叠加（支持居中或自定义偏移位置）。            |
| **Image Concat**                  | 水平或垂直拼接最多 5 张图片。                             |
| **Add Solid Color Background**    | 为透明图片添加纯色背景（支持 HEX 或 RGB 颜色指定）。     |
| **Add Solid Color Background V2** | 添加纯色背景并支持四边边距设置及自动正方形模式。          |
| **Solid Color Background**        | 自定义生成纯色背景图片（支持 RGB / RGBA / 灰度 L 模式）。 |

#### 1.3 图像缩放与调整尺寸
| 节点名称                       | 功能描述                                               |
| ------------------------------ | ------------------------------------------------------ |
| **Resize Image**               | 按指定宽高直接调整图片尺寸。                           |
| **Resize Image V2**            | 按固定宽或固定高等比例缩放图片。                       |
| **Resize Image By Ratio**      | 按比例系数缩放图片。                                   |
| **Resize Image By Max Size**   | 等比缩小图片，确保尺寸不超过指定最大值。               |
| **Resize Image By Min Size**   | 等比放大图片，确保尺寸不低于指定最小值。               |
| **Scale Image**                | 多模式缩放图片（AUTO / STRETCH / FILL / PAD）。         |
| **Upscale Image**              | 多模式放大图片，支持锐化处理（仅对放大操作生效）。     |

#### 1.4 图像裁剪
| 节点名称                   | 功能描述                                                        |
| -------------------------- | --------------------------------------------------------------- |
| **Crop Image**             | 按左/上/右/下边距裁剪图片。                                     |
| **Safe Image Crop**        | 从中心裁剪图片到最近指定倍数（如 64），确保与 VAE 空间兼容。   |
| **Bounding Box Cropping**  | 根据图片内透明对象的最小外接矩形自动裁剪，输出去除透明边缘的图片。 |

#### 1.5 蒙版操作
| 节点名称                          | 功能描述                                                 |
| --------------------------------- | -------------------------------------------------------- |
| **Image to Mask**                 | 将图片指定颜色通道（R/G/B/Alpha）转换为蒙版。           |
| **Mask to Image**                 | 将蒙版转换为灰度图像。                                   |
| **Invert Mask**                   | 反转蒙版中白色与黑色区域。                               |
| **Extract Transparent Image**     | 使用蒙版从原图中提取前景，输出带透明通道的抠图。         |
| **Mask Background to Transparent**| 将蒙版图片中非蒙版区域（白色以外的部分）转换为透明。     |
| **Mask Bounding Box Cropping**    | 根据蒙版有效区域的边界矩形裁剪蒙版图片。                 |
| **Morphological Transform**       | 对蒙版进行形态学操作（膨胀 / 腐蚀），可控制核大小和迭代次数。 |
| **Mask Contour Extractor**        | 提取蒙版轮廓并在图片上以指定颜色绘制轮廓线。             |

#### 1.6 颜色通道转换
| 节点名称          | 功能描述                       |
| ----------------- | ------------------------------ |
| **RGB to RGBA**   | 将 RGB 图片转换为 RGBA 格式。  |
| **RGBA to RGB**   | 将 RGBA 图片转换为 RGB 格式。  |

#### 1.7 图像信息获取
| 节点名称            | 功能描述                                   |
| ------------------- | ------------------------------------------ |
| **Get Image Size**  | 获取图片的宽、高及文件大小（字节数）。     |
| **Get Images Count**| 获取图片批次（batch）中的图片数量。        |
| **Get Side Length** | 获取图片最长边或最短边的像素长度。         |
| **Is Transparent**  | 检测图片是否包含透明区域（可设置阈值）。   |

---

### 二、文本处理  
| 节点名称                            | 功能描述                                               |
| ----------------------------------- | ------------------------------------------------------ |
| **Input Text**                      | 输入单段文本字符串。                                   |
| **Input Text List**                 | 输入最多 4 个文本字符串，组合为列表输出。              |
| **Connect Text Strings**            | 将列表与多个字符串按分隔符合并为一个字符串。           |
| **Simple Connect Text Strings**     | 将两个字符串按分隔符简单连接（支持列表批量处理）。     |
| **Text Replace**                    | 文本内容替换（支持普通匹配和正则表达式，可忽略大小写）。|
| **Show Text**                       | 在节点面板中显示文本内容。                             |
| **Show Anything**                   | 在节点面板中显示任意类型的数据（文本、数字、列表等）。 |

---

### 三、视频相关
| 节点名称                       | 功能描述                                          |
| ------------------------------ | ------------------------------------------------- |
| **Load Video**                 | 从视频 URL 下载视频并保存到指定目录，返回文件路径和预览 URL。 |
| **Preview Video**              | 输入视频 URL，在节点面板中嵌入预览播放器。        |
| **Save Video**                 | 将视频 URL 对应的视频文件保存到本地指定目录。     |
| **Save Video to Aliyun OSS**   | 将视频文件（本地路径或 URL）上传到阿里云 OSS。    |

---

### 四、数据处理  
| 节点名称                          | 功能描述                                                   |
| --------------------------------- | ---------------------------------------------------------- |
| **Hex to RGB**                    | 将十六进制颜色码转换为 R / G / B 三个整数值。              |
| **RGB to Hex**                    | 将 R / G / B 三个整数值转换为十六进制颜色码字符串。        |
| **Calculate**                     | 执行两个数值的加、减、乘、除、取余运算，同时输出字符串/整数/浮点数。 |
| **Convert Num Type**              | 将数值转换为整数 / 浮点数 / 字符串类型。                   |
| **KeyValue**                      | 从 JSON 字符串中按键路径提取值（支持点路径和数组索引）。   |
| **Serialize JSON Object**         | 将 JSON 对象序列化为 JSON 字符串。                         |
| **Deserialize JSON String**       | 将 JSON 字符串反序列化为 JSON 对象。                       |

---

### 五、实用工具  
| 节点名称                       | 功能描述                                                          |
| ------------------------------ | ----------------------------------------------------------------- |
| **PhantomTankEffect**          | 生成"幽灵坦克"（幻影坦克）效果图，可调节偏移量、灰度方法等参数。  |
| **Save to Aliyun OSS**         | 将文件（图片/视频等）上传到阿里云 OSS，支持签名 URL。             |
| **Save Metadata**              | 将元数据（workflow / prompt 等）写入 PNG 图片并保存。             |
| **Load Metadata From URL**     | 从指定 URL 下载图片并读取其中的元数据信息。                       |

---

## 🛠️ 开发与贡献  
- **问题反馈**：通过 [GitHub Issues](https://github.com/ihmily/ComfyUI-Light-Tool/issues) 提交。  
- **代码贡献**：欢迎 Fork 仓库并提交 Pull Request。  
- **文档更新**：若发现描述不准确或遗漏，请随时告知。 

---

## 📖 许可证  
本项目遵循 [MIT License](LICENSE)。  

---

