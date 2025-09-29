# coding:utf-8
"""
JM换脸节点 - ComfyUI自定义节点
基于火山引擎视觉AI API实现人脸替换功能
"""

import base64
import io
import torch
import numpy as np
from PIL import Image
import requests
import json
import os

# 尝试导入火山引擎SDK - 支持两种导入方式
try:
    # 尝试新版SDK
    from volcengine.visual.VisualService import VisualService
except ImportError:
    try:
        # 尝试旧版SDK
        from volcengine.visual import VisualService
    except ImportError:
        # 如果都导入失败，创建一个简单的错误类
        class VisualService:
            def __init__(self):
                raise ImportError("请安装火山引擎SDK: pip install volcengine")


class JMFaceSwapV2:
    """
    JM换脸节点 v2.0
    使用火山引擎API进行高质量人脸替换
    
    支持参数：
    - access_key: 火山引擎Access Key
    - secret_key: 火山引擎Secret Key
    - gpen: 高清效果 (0.0-1.0)
    - skin: 美化效果/肤色 (0.0-1.0)
    - keep_glass: 是否保留眼镜特征 (True/False)
    """
    
    def __init__(self):
        """初始化"""
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数类型"""
        return {
            "required": {
                "resource_face": ("IMAGE",),       # 原图像（需要被替换脸部的图像）
                "target_face": ("IMAGE",),         # 目标脸部图像（想要替换成的脸）
                "access_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入火山引擎Access Key"
                }),
                "secret_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入火山引擎Secret Key"
                }),
                "gpen": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "skin": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "keep_glass": ("BOOLEAN", {
                    "default": True,
                    "label_on": "保留眼镜",
                    "label_off": "不保留眼镜"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("换脸结果",)
    FUNCTION = "face_swap"
    CATEGORY = "JM/图像处理"
    
    def tensor_to_base64(self, tensor):
        """
        将PyTorch张量转换为base64编码的图像，严格按照火山引擎API要求处理
        
        要求：
        - 单张图片base64转码后小于5MB
        - 图片尺寸：小于2048x2048（确保清晰度），大于64x64像素
        - 两张图片总大小小于8MB
        - 建议使用JPG格式
        
        Args:
            tensor: 形状为 (B, H, W, C) 的图像张量
            
        Returns:
            str: base64编码的图像字符串
        """
        # 确保张量是正确的形状 (B, H, W, C)
        if len(tensor.shape) == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # 移除批次维度
        
        # 转换为numpy数组
        image_np = tensor.detach().cpu().numpy()
        
        # 确保数值范围在0-255之间
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size
        
        print(f"📏 原始图像尺寸: {width}x{height}")
        
        # 1. 检查最小尺寸限制 (64x64)
        if width < 64 or height < 64:
            # 放大到最小尺寸
            scale = max(64 / width, 64 / height)
            new_width = max(64, int(width * scale))
            new_height = max(64, int(height * scale))
            print(f"🔍 图像过小，放大到: {new_width}x{new_height}")
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = new_width, new_height
        
        # 2. 检查最大尺寸限制 (2048x2048 以确保清晰度)
        max_dimension = 2048
        if width > max_dimension or height > max_dimension:
            # 缩小到最大尺寸
            scale = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"🔄 图像过大，压缩到: {new_width}x{new_height} (保持清晰度)")
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = new_width, new_height
        
        # 3. 尝试不同的JPEG质量设置，确保base64编码后小于5MB
        max_base64_size = 5 * 1024 * 1024  # 5MB
        
        # 优先保证清晰度，从较高质量开始
        for quality in [90, 85, 80, 75, 70, 65, 60, 55]:
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 检查base64大小
            base64_size = len(image_base64)
            size_mb = base64_size / (1024 * 1024)
            
            print(f"📦 质量{quality}%, base64大小: {size_mb:.2f}MB")
            
            if base64_size < max_base64_size:
                print(f"✅ 符合要求！最终尺寸: {width}x{height}, 质量: {quality}%, 大小: {size_mb:.2f}MB")
                return image_base64
        
        # 如果所有质量都不满足，进一步缩小图像
        print("⚠️ 需要进一步缩小图像尺寸")
        for scale_factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # 确保不小于最小尺寸
            if new_width < 64 or new_height < 64:
                continue
                
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            resized_image.save(buffer, format='JPEG', quality=80, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            base64_size = len(image_base64)
            size_mb = base64_size / (1024 * 1024)
            
            print(f"📦 缩放{scale_factor}倍, 尺寸: {new_width}x{new_height}, 大小: {size_mb:.2f}MB")
            
            if base64_size < max_base64_size:
                print(f"✅ 符合要求！最终尺寸: {new_width}x{new_height}, 大小: {size_mb:.2f}MB")
                return image_base64
        
        # 如果还是不行，使用保底质量
        print("⚠️ 使用保底质量压缩")
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=55, optimize=True)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        base64_size = len(image_base64)
        size_mb = base64_size / (1024 * 1024)
        print(f"📦 最终大小: {size_mb:.2f}MB")
        
        return image_base64
    
    def _compress_for_total_limit(self, tensor, target_size=3.5):
        """
        为了满足总体8MB限制而进行的额外压缩
        
        Args:
            tensor: 图像张量
            target_size: 目标大小(MB)，默认3.5MB
            
        Returns:
            str: 压缩后的base64字符串
        """
        # 确保张量是正确的形状
        if len(tensor.shape) == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        
        # 转换为numpy数组
        image_np = tensor.detach().cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size
        
        target_base64_size = target_size * 1024 * 1024  # 转换为字节
        
        # 先尝试不同的质量设置，保持一定清晰度
        for quality in [75, 70, 65, 60, 55, 50]:
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if len(image_base64) < target_base64_size:
                size_mb = len(image_base64) / (1024 * 1024)
                print(f"✅ 质量{quality}%满足要求，大小: {size_mb:.2f}MB")
                return image_base64
        
        # 如果质量压缩还不够，缩小尺寸
        for scale_factor in [0.8, 0.7, 0.6, 0.5, 0.4]:
            new_width = max(64, int(width * scale_factor))
            new_height = max(64, int(height * scale_factor))
            
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            resized_image.save(buffer, format='JPEG', quality=60, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if len(image_base64) < target_base64_size:
                size_mb = len(image_base64) / (1024 * 1024)
                print(f"✅ 缩放{scale_factor}倍满足要求，尺寸: {new_width}x{new_height}, 大小: {size_mb:.2f}MB")
                return image_base64
        
        # 最后的保底处理，保持基本清晰度
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=45, optimize=True)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        size_mb = len(image_base64) / (1024 * 1024)
        print(f"⚠️ 使用保底质量压缩，大小: {size_mb:.2f}MB")
        
        return image_base64
    
    def load_config(self):
        """从配置文件加载API密钥（作为备用方案）"""
        try:
            # 获取当前文件所在目录的上级目录
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(current_dir, 'API key_config.json')
            
            if not os.path.exists(config_path):
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            volcengine_config = config.get('volcengine', {})
            access_key = volcengine_config.get('access_key')
            secret_key = volcengine_config.get('secret_key')
            
            if not access_key or not secret_key:
                return None
            
            return access_key, secret_key
            
        except Exception as e:
            return None
    
    def base64_to_tensor(self, base64_str):
        """
        将base64编码的图像转换为PyTorch张量
        
        Args:
            base64_str: base64编码的图像字符串
            
        Returns:
            torch.Tensor: 形状为 (1, H, W, C) 的图像张量
        """
        # 解码base64
        image_data = base64.b64decode(base64_str)
        
        # 转换为PIL图像
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 确保是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为numpy数组
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        
        # 转换为PyTorch张量并添加批次维度
        tensor = torch.from_numpy(image_np).unsqueeze(0)  # (1, H, W, C)
        
        return tensor
    
    def face_swap(self, resource_face, target_face, access_key, secret_key, gpen, skin, keep_glass):
        """
        执行人脸替换
        
        Args:
            resource_face: 原图像张量 (B, H, W, C) - 需要被替换脸部的图像
            target_face: 目标脸部图像张量 (B, H, W, C) - 想要替换成的脸
            access_key: 火山引擎Access Key
            secret_key: 火山引擎Secret Key
            gpen: GPEN参数 (0.0-1.0) - 高清效果
            skin: 肤色参数 (0.0-1.0) - 美化效果（肤色）
            keep_glass: 是否保留眼镜 (True/False) - 输出图中是否保留用户图中的眼镜特征
            
        Returns:
            tuple: 包含结果图像张量的元组
        """
        try:
            # 初始化视觉服务
            visual_service = VisualService()
            
            # 优先使用节点输入的API密钥
            if access_key and secret_key:
                visual_service.set_ak(access_key)
                visual_service.set_sk(secret_key)
                print("✅ 使用节点输入的API密钥")
            else:
                # 备用方案：从配置文件加载
                config_result = self.load_config()
                if config_result is None:
                    print("❌ 无法获取API密钥，返回原图像")
                    return (resource_face,)
                
                access_key, secret_key = config_result
                visual_service.set_ak(access_key)
                visual_service.set_sk(secret_key)
                print("✅ 使用配置文件中的API密钥")
            
        except Exception as e:
            print(f"❌ 初始化火山引擎服务失败: {e}")
            print("💡 请确保已安装火山引擎SDK: pip install volcengine")
            return (resource_face,)
        
        try:
            # 转换输入图像为base64
            print("🔄 处理目标脸部图像...")
            target_b64 = self.tensor_to_base64(target_face)
            
            print("🔄 处理原图像...")
            source_b64 = self.tensor_to_base64(resource_face)
            
            # 检查总体大小限制 (8MB)
            total_size = len(target_b64) + len(source_b64)
            total_size_mb = total_size / (1024 * 1024)
            print(f"📊 两张图片总大小: {total_size_mb:.2f}MB")
            
            if total_size_mb > 8:
                print("⚠️ 两张图片总大小超过8MB限制，需要进一步压缩...")
                # 重新处理，使用更严格的压缩
                print("🔄 重新压缩目标脸部图像...")
                target_b64 = self._compress_for_total_limit(target_face, target_size=3.5)
                
                print("🔄 重新压缩原图像...")
                source_b64 = self._compress_for_total_limit(resource_face, target_size=3.5)
                
                # 再次检查
                total_size = len(target_b64) + len(source_b64)
                total_size_mb = total_size / (1024 * 1024)
                print(f"📊 压缩后总大小: {total_size_mb:.2f}MB")
            
            # 准备API请求参数
            # 注意：根据API文档，第一张图是目标脸部，第二张图是需要被替换的原图
            form_data = {
                "req_key": "faceswap_ai",
                "binary_data_base64": [target_b64, source_b64],
                "do_risk": True,
                "gpen": float(gpen),
                "skin": float(skin),
                "keep_glass": bool(keep_glass),
                "return_url": True,
                "logo_info": {
                    "add_logo": False,  # 关闭水印
                    "position": 1,
                    "language": 0,
                    "opacity": 0.0,
                    "logo_text_content": ""
                }
            }
            
            # 调用API
            print(f"🔄 正在进行人脸替换...")
            print(f"📊 参数设置: GPEN={gpen}, Skin={skin}, Keep_Glass={keep_glass}")
            print(f"📊 目标脸部图像尺寸: {target_face.shape}")
            print(f"📊 原图像尺寸: {resource_face.shape}")
            
            response = visual_service.cv_process(form_data)
            print(f"🔍 API响应: {response}")
            
            # 处理响应
            if isinstance(response, dict) and response.get('code') == 10000:
                data = response.get('data', {})
                
                # 优先使用base64数据
                if 'binary_data_base64' in data and data['binary_data_base64']:
                    result_b64 = data['binary_data_base64'][0]
                    result_tensor = self.base64_to_tensor(result_b64)
                    print("✅ 人脸替换成功！")
                    return (result_tensor,)
                
                # 如果没有base64数据，尝试从URL下载
                elif 'image_urls' in data and data['image_urls']:
                    image_url = data['image_urls'][0]
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        # 将下载的图像转换为base64再转换为张量
                        image_b64 = base64.b64encode(img_response.content).decode('utf-8')
                        result_tensor = self.base64_to_tensor(image_b64)
                        print("✅ 人脸替换成功！(通过URL)")
                        return (result_tensor,)
                    else:
                        raise Exception("无法下载结果图像")
                
                else:
                    raise Exception("API响应中没有找到图像数据")
            
            else:
                error_msg = f"API调用失败: {response}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"❌ 人脸替换失败: {str(e)}")
            # 返回原图像作为错误处理
            return (resource_face,)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "JM:face swap_v2.0": JMFaceSwapV2
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JM:face swap_v2.0": "JM:face swap_v2.0"
} 