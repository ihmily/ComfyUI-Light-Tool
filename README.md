# ComfyUI-Light-Tool  

简体中文 / [English](./README_EN.md)

大量轻量实用的 ComfyUI 自定义工具节点，提升工作流效率与灵活性。

---

## 📦 安装指南  
### 1. 手动安装  
```bash
# 进入 ComfyUI 的 custom_nodes 目录
cd ComfyUI/custom_nodes

# 克隆本仓库
git clone https://github.com/ihmily/ComfyUI-Light-Tool.git

# 重启 ComfyUI
```

### 2. 使用 ComfyUI-Manager 安装  
通过 ComfyUI-Manager 自动安装：  
1. 进入 ComfyUI 界面的 **Manager** 面板。  
2. 在 **Custom Nodes** 标签下搜索 `ComfyUI-Light-Tool`。  
3. 点击 **Install** 并重启 ComfyUI。

---

## 🎨 功能分类与节点列表  

### 一、图像处理  
| 节点名称                                                     | 功能描述                                 |
| ------------------------------------------------------------ | ---------------------------------------- |
| **Load Image**                                               | 加载本地图片文件（支持保持RGBA透明度）。 |
| **Load Image From URL**                                      | 从 URL 下载并加载图片。                  |
| **Load Image List**                                          | 批量加载目录中的图片。                   |
| **Image Overlay**                                            | 合并两张图片（需蒙版和图片尺寸一致）。   |
| **Simple Image Overlay**                                     | 任意图片叠加（支持调整位置）。           |
| **Image Concat**                                             | 水平或垂直拼接多张图片。                 |
| **Resize Image** / **Resize Image V2** / **Resize by Ratio/Max Size** | 灵活调整图片尺寸、比例或最大尺寸。       |
| **Scale Image**                                              | 使用算法缩放图片。                       |
| **Upscale Image**                                            | 使用算法放大图片。                       |
| **Add Solid Color Background**                               | 为透明图片添加纯色背景。                 |
| **Image to Mask** / **Mask to Image**                        | 图片与蒙版的相互转换。                   |
| **Invert Mask**                                              | 反转蒙版颜色。                           |
| **Bounding Box Cropping**                                    | 根据坐标裁剪图片。                       |
| **RGB to RGBA** / **RGBA to RGB**                            | 转换图片颜色通道。                       |
| **Morphological Transform**                                  | 对蒙版进行形态学操作（如腐蚀、膨胀）。   |
| **Extract Transparent Image**                                | 使用蒙版从原图中提取前景透明抠图         |
| **Mask Background to Transparent**                           | 转换蒙版图片中的非蒙版部分为透明         |

---

### 二、文本处理  
| 节点名称                                                   | 功能描述             |
| ---------------------------------------------------------- | -------------------- |
| **Input Text**                                             | 输入文本。           |
| **Input Text List**                                        | 输入文本列表。       |
| **Connect Text Strings** / **Simple Connect Text Strings** | 合并多个文本字符串。 |
| **Show Text**                                              | 显示文本内容。       |

---

### 三、视频相关
| 节点名称          | 功能描述                      |
| ----------------- | ----------------------------- |
| **Load Video**    | 加载视频（从视频地址）。      |
| **Preview Video** | 预览视频内容 （从视频地址）。 |
| **Save Video**    | 将视频保存到本地特定目录。    |

---

### 四、数据处理  
| 节点名称                       | 功能描述                             |
| ------------------------------ | ------------------------------------ |
| **Get Image Size**             | 获取图片的宽高和文件大小信息。       |
| **Hex to RGB**                 | 将十六进制颜色码转换为 RGB 值。      |
| **Calculate**                  | 执行数值计算（如加减乘除、百分比）。 |
| **Convert Num Type**           | 转换数值类型（整数/浮点数/字符串）。 |
| **Get Images Count**           | 统计图片列表数量。                   |
| **KeyValue**                   | 从 JSON 数据中提取键值对。           |
| **Serialize/Deserialize JSON** | 序列化或反序列化 JSON 对象。         |

---

### 五、实用工具  
| 艺术名称                                       | 功能描述                               |
| ---------------------------------------------- | -------------------------------------- |
| **PhantomTankEffect**                          | 生成“幽灵坦克” ”幻影坦克“ 效果         |
| **Is Transparent**                             | 检测图片是否是透明的。                 |
| **Mask Bounding Box Cropping**                 | 根据蒙版边界裁剪图片。                 |
| **Save to Aliyun OSS**                         | 上传(图片/视频)到阿里云 OSS 存储。     |
| **Save Metadata** / **Load Metadata From URL** | 保存或加载元数据（如图片描述、参数）。 |
| **SolidColorBackground**                       | 自定义生成纯色背景图片                 |

---

## 🛠️ 开发与贡献  
- **问题反馈**：通过 [GitHub Issues](https://github.com/ihmily/ComfyUI-Light-Tool/issues) 提交。  
- **代码贡献**：欢迎 Fork 仓库并提交 Pull Request。  
- **文档更新**：若发现描述不准确或遗漏，请随时告知。 

---

## 📖 许可证  
本项目遵循 [MIT License](LICENSE)。  

---

