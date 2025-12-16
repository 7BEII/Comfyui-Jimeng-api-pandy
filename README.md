# ComfyUI-Jimeng-API-Pandy

基于火山引擎视觉AI API的ComfyUI自定义节点集合，提供高质量的人脸替换功能。

## 功能特性

### JM换脸节点 v2.0
- 🎭 **高质量人脸替换**：基于火山引擎视觉AI API实现
- 🔧 **灵活配置**：支持直接在节点中输入API密钥或使用配置文件
- 🎨 **参数调节**：支持GPEN高清效果、肤色美化、眼镜保留等参数
- 📏 **智能压缩**：自动处理图像尺寸和大小限制，确保API调用成功
- 🛡️ **错误处理**：完善的错误处理和降级机制

## 安装要求

### 环境要求
- Python 3.8+
- ComfyUI
- PyTorch
- 火山引擎SDK

### 依赖安装
```bash
# 激活conda环境
conda activate yourvenv

# 安装所有依赖（推荐方式）
pip install -r requirements.txt

# 或者单独安装主要依赖
pip install volcengine>=1.0.0
pip install requests>=2.25.1
pip install Pillow>=8.0.0
pip install numpy>=1.19.0
pip install torch>=1.7.0
```

### 依赖包说明
- **volcengine**: 火山引擎SDK，用于调用视觉AI API
- **requests**: HTTP请求库，用于API调用和图像下载
- **Pillow**: 图像处理库，用于图像格式转换和压缩
- **numpy**: 数值计算库，用于数组操作
- **torch**: PyTorch深度学习框架，用于张量操作
- **xlrd**: Excel文件读取库（其他节点可能需要）

## 使用方法

### 1. JM换脸节点 v2.0

#### 输入参数
- **resource_face** (IMAGE): 原图像，需要被替换脸部的图像
- **target_face** (IMAGE): 目标脸部图像，想要替换成的脸
- **access_key** (STRING): 火山引擎Access Key（可选，支持配置文件备用）
- **secret_key** (STRING): 火山引擎Secret Key（可选，支持配置文件备用）
- **gpen** (FLOAT): 高清效果参数 (0.0-1.0，默认0.8)
- **skin** (FLOAT): 美化效果/肤色参数 (0.0-1.0，默认0.1)
- **keep_glass** (BOOLEAN): 是否保留眼镜特征 (默认True)

#### 输出
- **换脸结果** (IMAGE): 处理后的图像张量，形状为 (1, H, W, C)

#### API密钥配置方式

**方式1：直接在节点中输入（推荐）**
- 在节点的access_key和secret_key字段中直接输入你的火山引擎API密钥
- 这种方式更安全，不会在配置文件中保存敏感信息

**方式2：使用配置文件（备用）**
- 在项目根目录创建 `API key_config.json` 文件
- 文件格式：
```json
{
    "volcengine": {
        "access_key": "你的Access Key",
        "secret_key": "你的Secret Key"
    }
}
```

### 2. 图像处理要求

节点会自动处理以下限制：
- 单张图片base64转码后小于5MB
- 图片尺寸：64x64 到 2048x2048 像素
- 两张图片总大小小于8MB
- 自动压缩和格式转换

### 3. 使用示例

1. 在ComfyUI中添加JM换脸节点
2. 连接原图像到 `resource_face` 输入
3. 连接目标脸部图像到 `target_face` 输入
4. 输入你的火山引擎API密钥（或确保配置文件存在）
5. 调整参数（可选）
6. 运行工作流

## 技术特性

### 张量格式
- 输入图像张量形状：`(B, H, W, C)` 其中B=1
- 输出图像张量形状：`(1, H, W, C)`
- 数值范围：0.0-1.0 (float32)

### 图像压缩策略
1. **尺寸检查**：确保图像在64x64到2048x2048范围内
2. **质量压缩**：从90%质量开始，逐步降低到55%
3. **尺寸压缩**：如果质量压缩不够，进一步缩小图像尺寸
4. **总体限制**：确保两张图片总大小不超过8MB

### 错误处理
- API密钥验证
- 网络请求重试
- 图像格式转换
- 降级处理（返回原图像）

## 火山引擎API文档

详细API文档请参考：[火山引擎视觉AI API文档](https://www.volcengine.com/docs/82379/1399008#99a7c9ca)

## 更新日志


### v2.1 (2024-12-16)
- 🐛 **修复 Seedream 节点图片传入问题**：修复了从其他节点（如 PDIMAGE_LongerSize）传入图片时无法正确识别的问题
  - 原因：	ensor_to_base64 函数使用 image_np.max() <= 1.0 判断图片值范围不可靠，当图片接近全黑时会误判
  - 修复：改为检查数据类型 (
p.float32/
p.float64) 并添加 
p.clip 确保值在有效范围内

### v2.0 (当前版本)
- ✅ 支持直接在节点中输入API密钥
- ✅ 保留配置文件作为备用方案
- ✅ 优化图像压缩算法
- ✅ 改进错误处理机制
- ✅ 更新参数名称（resource_face, target_face）

### v1.0
- 基础人脸替换功能
- 配置文件API密钥管理
- 基本图像处理

## 故障排除

### 常见问题

1. **"请安装火山引擎SDK"错误**
   ```bash
   pip install volcengine
   ```

2. **"无法获取API密钥"错误**
   - 检查节点中的API密钥输入
   - 或检查配置文件是否存在且格式正确

3. **"图像过大"错误**
   - 节点会自动压缩图像
   - 如果仍然失败，请使用更小的图像

4. **API调用失败**
   - 检查网络连接
   - 验证API密钥是否正确
   - 检查火山引擎账户余额

### 调试信息
节点会输出详细的调试信息，包括：
- 图像尺寸和压缩信息
- API调用状态
- 错误详情

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
