import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
import math

class JMSEEDreamT2INode:
    """
    调用火山引擎即梦API生成图片的节点
    使用 doubao-seedream-3.0-t2i 模型
    """
    
    # 预定义的模型ID选项
    MODEL_IDS = {
        "doubao-seedream-3.0-t2i (基础模型)": "doubao-seedream-3-0-t2i-250415"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一只可爱的猫咪", 
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "提示词，支持接入文本节点"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的火山引擎API Key"
                }),
                "model_selection": (list(cls.MODEL_IDS.keys()), {
                    "default": "doubao-seedream-3.0-t2i (基础模型)"
                }),
                "aspect_ratio": (["1:1", "3:4", "4:3", "16:9", "9:16", "21:9"], {
                    "default": "3:4"
                }),
                # 修改：去掉了3K和4K，因为API限制最大边长为2048
                "image_size": (["1K", "2K"], {
                    "default": "2K"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "true",
                    "label_off": "false",
                    "tooltip": "是否在生成的图片中添加水印标识"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = """使用火山引擎即梦API生成图片。
    
模型：doubao-seedream-3.0-t2i
需要提供有效的API Key才能使用。

参数说明：
- prompt: 图片生成的提示词 (支持外部输入)
- aspect_ratio: 画幅比例
- image_size: 图片分辨率大小 (1K=1024px, 2K=2048px)
- seed: 随机种子
"""

    def generate_image(self, prompt, api_key, model_selection="doubao-seedream-3.0-t2i (基础模型)", 
                      aspect_ratio="3:4", image_size="2K", seed=-1, watermark=False):
        """调用火山引擎API生成图片"""
        
        # 优先读取apikey.txt
        apikey_path = os.path.join(os.path.dirname(__file__), "apikey.txt")
        file_api_key = ""
        if os.path.exists(apikey_path):
            with open(apikey_path, "r", encoding="utf-8") as f:
                file_api_key = f.read().strip()
        
        use_api_key = file_api_key if file_api_key else api_key
        if not use_api_key:
            raise ValueError("请在apikey.txt或前端页面输入有效的API Key")
        api_key = use_api_key
        
        # 设置默认值
        response_format = "url"
        api_endpoint = "https://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3"
        debug_mode = False
        n = 1  # 默认生成1张图片
        
        # 确定使用的模型ID
        model_id = self.MODEL_IDS[model_selection]
        
        # --- 处理尺寸逻辑 (aspect_ratio + image_size -> WxH) ---
        # 定义长边基准 (API限制最大2048)
        base_pixels = {
            "1K": 1024,
            "2K": 2048
        }
        
        # 获取长边长度，默认为2048
        long_edge = base_pixels.get(image_size, 2048)
        
        # 解析比例
        try:
            w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
        except:
            w_ratio, h_ratio = 3, 4 # 默认 fallback
            
        # 计算具体宽高
        if w_ratio == h_ratio:
            width, height = long_edge, long_edge
        elif w_ratio > h_ratio:
            width = long_edge
            height = int(long_edge * (h_ratio / w_ratio))
        else:
            height = long_edge
            width = int(long_edge * (w_ratio / h_ratio))
            
        # 确保宽高是8的倍数（避免潜在的对齐问题）
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        actual_size = f"{width}x{height}"
        
        if debug_mode:
            print(f"计算分辨率: {aspect_ratio} @ {image_size} -> {actual_size}")
        # ----------------------------------------------------
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 准备请求数据
        data = {
            "model": model_id,
            "prompt": prompt,
            "size": actual_size,
            "n": n,
            "response_format": response_format,
            "watermark": watermark
        }
        
        # 处理seed参数
        if seed != -1:
            if seed < 0: seed = 0
            elif seed > 2147483647: seed = seed % 2147483648
            if seed >= 0: data["seed"] = int(seed)
        
        if debug_mode:
            print(f"API请求数据: {json.dumps(data, ensure_ascii=False)}")
        
        try:
            # 发送请求
            response = requests.post(
                f"{api_endpoint}/images/generations",
                headers=headers,
                json=data,
                timeout=300
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {json.dumps(error_data['error'], ensure_ascii=False)}"
                except:
                    error_msg += f" - {response.text}"
                raise RuntimeError(error_msg)
            
            response.raise_for_status()
            result = response.json()
            
            # 处理返回的图片
            images = []
            if "data" in result:
                for img_data in result["data"]:
                    if response_format == "url":
                        img_url = img_data.get("url")
                        if img_url:
                            img_response = requests.get(img_url, timeout=300)
                            img = Image.open(BytesIO(img_response.content))
                    else:
                        import base64
                        b64_string = img_data.get("b64_json")
                        if b64_string:
                            img = Image.open(BytesIO(base64.b64decode(b64_string)))
                    
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(torch.from_numpy(img_array))
            
            if not images:
                raise ValueError("API未返回任何图片")
            
            output_images = torch.stack(images, dim=0)
            return (output_images,)
            
        except Exception as e:
            raise RuntimeError(f"生成图片时出错: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "JM:Seeddream 3.0_t2i": JMSEEDreamT2INode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JM:Seeddream 3.0_t2i": "JM:Seeddream 3.0_t2i"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]