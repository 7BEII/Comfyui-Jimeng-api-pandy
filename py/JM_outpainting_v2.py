# coding:utf-8
"""
JM Outpainting v2.0节点 - ComfyUI自定义节点
基于火山引擎视觉AI API实现图像扩展功能
"""

import base64
import io
import torch
import numpy as np
from PIL import Image
import requests
import random
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


class JMOutpaintingV2:
    """
    JM Outpainting v2.0节点
    使用火山引擎API进行图像扩展
  
    支持参数：
    - access_key: 火山引擎Access Key
    - secret_key: 火山引擎Secret Key
    - custom_prompt: 自定义提示词
    - scale: 缩放比例
    - seed: 随机种子
    - steps: 采样步数
    - strength: 强度
    - top/bottom/left/right: 扩展方向参数
    - max_height/max_width: 最大尺寸限制
    """
  
    def __init__(self):
        """初始化"""
        pass
  
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数类型"""
        return {
            "required": {
                "input_image": ("IMAGE",),    # 输入图像（RGB三通道）
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
                "custom_prompt": ("STRING", {
                    "default": "蓝色的海洋",
                    "multiline": True,
                    "placeholder": "输入扩展描述..."
                }),
                "top": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "bottom": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "left": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "right": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "max_height": ("INT", {
                    "default": 1920,
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
                "max_width": ("INT", {
                    "default": 1920,
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
  
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("扩展图像",)
    FUNCTION = "outpaint_image"
    CATEGORY = "JM/图像扩展"

    def tensor_to_base64(self, tensor):
        """
        将PyTorch张量转换为base64编码的图像
      
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
      
        # 确保是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
      
        # 转换为base64，使用JPEG格式
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95, optimize=True)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
      
        # 检查base64大小
        size_mb = len(image_base64) / (1024 * 1024)
      
        return image_base64
  
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

    def outpaint_image(self, input_image, access_key, secret_key, custom_prompt, top, bottom, left, right, 
                      scale, steps, strength, max_height, max_width, seed):
        """
        执行图像扩展
      
        Args:
            input_image: 输入图像张量 (B, H, W, C)
            access_key: 火山引擎Access Key
            secret_key: 火山引擎Secret Key
            custom_prompt: 自定义提示词
            top: 向上扩展比例
            bottom: 向下扩展比例
            left: 向左扩展比例
            right: 向右扩展比例
            scale: 缩放比例
            steps: 采样步数
            strength: 强度
            max_height: 最大高度
            max_width: 最大宽度
            seed: 随机种子
          
        Returns:
            tuple: 包含结果图像张量的元组
        """
        
        # 检查API密钥是否为空
        if not access_key.strip() or not secret_key.strip():
            raise ValueError("请在界面输入火山引擎Access Key和Secret Key")
        
        try:
            # 初始化视觉服务
            visual_service = VisualService()
            visual_service.set_ak(access_key.strip())
            visual_service.set_sk(secret_key.strip())
            
            # 转换输入图像为base64
            input_image_b64 = self.tensor_to_base64(input_image)
          
            # 处理随机种子
            if seed == -1:
                seed = random.randint(0, 2147483647)
          
            # 准备API请求参数
            form_data = {
                "req_key": "i2i_outpainting",
                "custom_prompt": str(custom_prompt),
                "binary_data_base64": [input_image_b64],
                "scale": float(scale),
                "seed": int(seed),
                "steps": int(steps),
                "strength": float(strength),
                "top": float(top),
                "bottom": float(bottom),
                "left": float(left),
                "right": float(right),
                "max_height": int(max_height),
                "max_width": int(max_width)
            }
          
            # 调用API
            response = visual_service.cv_process(form_data)
          
            # 处理响应
            if isinstance(response, dict) and response.get('code') == 10000:
                data = response.get('data', {})
              
                # 优先使用base64数据
                if 'binary_data_base64' in data and data['binary_data_base64']:
                    result_b64 = data['binary_data_base64'][0]
                    result_tensor = self.base64_to_tensor(result_b64)
                    print("✅ Outpainting扩展成功完成")
                    return (result_tensor,)
              
                # 如果没有base64数据，尝试从URL下载
                elif 'image_urls' in data and data['image_urls']:
                    image_url = data['image_urls'][0]
                  
                    # 尝试多种网络配置下载图像
                    # 方式1: 直接下载（无代理）
                    try:
                        session = requests.Session()
                        session.trust_env = False  # 忽略环境变量中的代理设置
                        img_response = session.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            image_b64 = base64.b64encode(img_response.content).decode('utf-8')
                            result_tensor = self.base64_to_tensor(image_b64)
                            print("✅ Outpainting扩展成功完成")
                            return (result_tensor,)
                    except Exception as e:
                        pass
                  
                    # 方式2: 使用系统代理
                    try:
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            image_b64 = base64.b64encode(img_response.content).decode('utf-8')
                            result_tensor = self.base64_to_tensor(image_b64)
                            print("✅ Outpainting扩展成功完成")
                            return (result_tensor,)
                    except Exception as e:
                        pass
                  
                    # 方式3: 尝试不同的User-Agent
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        session = requests.Session()
                        session.trust_env = False
                        img_response = session.get(image_url, headers=headers, timeout=30)
                        if img_response.status_code == 200:
                            image_b64 = base64.b64encode(img_response.content).decode('utf-8')
                            result_tensor = self.base64_to_tensor(image_b64)
                            print("✅ Outpainting扩展成功完成")
                            return (result_tensor,)
                    except Exception as e:
                        pass
                  
                    # 如果所有方式都失败
                    raise Exception("所有下载方式都失败，请检查网络连接")
              
                else:
                    raise Exception("API响应中没有找到图像数据")
          
            else:
                error_msg = f"API调用失败: {response}"
                raise Exception(error_msg)
              
        except Exception as e:
            print(f"❌ Outpainting扩展失败: {str(e)}")
            # 返回原图像作为错误处理
            return (input_image,)


NODE_CLASS_MAPPINGS = {
    "JM:Outpainting_v2.0": JMOutpaintingV2
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JM:Outpainting_v2.0": "JM:Outpainting_v2.0"
}