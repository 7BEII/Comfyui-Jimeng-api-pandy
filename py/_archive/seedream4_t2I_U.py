import os
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import requests
import urllib3

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Seeddream_Universal_T2I:
    """
    JM:Seedream 通用文生图节点 (T2I)
    支持 Seedream 4.0 / 4.5 模型切换
    支持 1K/2K/3K/4K (Base-1024) 分辨率与比例控制
    """
    
    # 定义支持的模型列表
    MODEL_MAP = {
        "Seedream 4.5 (doubao-seedream-4-5-251128)": "doubao-seedream-4-5-251128",
        "Seedream 4.0 (doubao-seedream-4-0-250828)": "doubao-seedream-4-0-250828"
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "sk-xxx (必填，或从环境变量读取)"
                }),
                "model": (list(cls.MODEL_MAP.keys()), {
                    "default": "Seedream 4.5 (doubao-seedream-4-5-251128)",
                    "tooltip": "选择即梦(Seedream)模型版本"
                }),
                "prompt": ("STRING", {
                    "default": "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车...", 
                    "multiline": True
                }),
                # === 新增比例控制 (移除了无用的 auto) ===
                "aspect_ratio": (["1:1", "3:4", "4:3", "16:9", "9:16", "21:9"], {
                    "default": "3:4"
                }),
                # === 更新分辨率定义 (含3K) ===
                "size": (["1K", "2K", "3K", "4K"], {"default": "2K"}),
                "watermark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "✨即梦AI生成"

    def get_dimensions(self, aspect_ratio, size_key):
        """计算目标宽高，包含防超标逻辑 (T2I版)"""
        
        # API 严格上限 (4096*4096)
        MAX_PIXELS = 16777216
        
        # === 像素定义 (Base-1024) ===
        pixel_counts = {
            "1K": 1024 * 1024,      # ~100万像素
            "2K": 2048 * 2048,      # ~420万像素
            "3K": 3072 * 3072,      # ~940万像素
            "4K": 4096 * 4096       # ~1677万像素 (硬上限)
        }
        
        target_pixels = pixel_counts.get(size_key, 2048*2048)
        
        # === 比例处理 ===
        ratios = {
            "1:1": (1, 1),
            "3:4": (3, 4), "4:3": (4, 3), 
            "16:9": (16, 9), "9:16": (9, 16), 
            "21:9": (21, 9)
        }
        w_ratio, h_ratio = ratios.get(aspect_ratio, (3, 4))

        # === 核心计算 ===
        ratio_val = w_ratio / h_ratio
        
        # H = sqrt(Area / Ratio)
        h = (target_pixels / ratio_val) ** 0.5
        w = h * ratio_val
        
        # 对齐 64 (向上取整)
        w = int(((w + 63) // 64) * 64)
        h = int(((h + 63) // 64) * 64)
        
        # === 安全检查与修正 (防4K溢出) ===
        # 如果总像素超过限制，循环减少尺寸直到合规
        while w * h > MAX_PIXELS:
            if w > h:
                w -= 64
            else:
                h -= 64
            if w < 64 or h < 64: break # 保底
            
        return f"{w}x{h}", w, h

    def generate_image(self, api_key, model, prompt, aspect_ratio, size, watermark):
        # 1. 基础校验
        if not api_key:
            api_key = os.environ.get("ARK_API_KEY")
        
        if not api_key:
            raise ValueError("❌ 错误：API Key 不能为空！")
            
        # 获取模型ID
        model_id = self.MODEL_MAP.get(model, "doubao-seedream-4-5-251128")
        
        # === 计算实际分辨率 ===
        size_str, w, h = self.get_dimensions(aspect_ratio, size)
        
        # 2. 准备请求
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 文生图 Payload
        payload = {
            "model": model_id,
            "prompt": prompt,
            "size": size_str, # 使用计算出的 "WxH" 字符串
            "sequential_image_generation": "disabled",
            "response_format": "b64_json",
            "stream": False,
            "watermark": watermark
        }

        print(f"🚀 [JM:Seedream T2I] 发送请求... 模型: {model_id}")
        print(f"📐 规格: {size} ({aspect_ratio}) -> 实际尺寸: {size_str} (像素: {w*h})")

        # 3. 发送请求 (抗网络干扰)
        try:
            session = requests.Session()
            session.trust_env = False # 强制直连，忽略系统代理
            
            adapter = requests.adapters.HTTPAdapter(max_retries=3)
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            response = session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=120, # 4K 生成可能需要较长时间
                verify=False
            )
            
            if response.status_code != 200:
                if "size" in response.text:
                    raise RuntimeError(f"❌ 分辨率报错: {response.text}")
                raise RuntimeError(f"❌ API 请求失败 (状态码 {response.status_code}):\n{response.text}")

            # 4. 解析结果
            res_json = response.json()
            
            if "data" in res_json and len(res_json["data"]) > 0:
                b64_data = res_json["data"][0].get("b64_json")
                if not b64_data:
                     # 兼容 URL 模式
                     image_url = res_json["data"][0].get("url")
                     if image_url:
                         print(f"📥 下载图片: {image_url}")
                         img_resp = session.get(image_url, timeout=60, verify=False)
                         img = Image.open(BytesIO(img_resp.content))
                     else:
                         raise RuntimeError("SDK 返回数据异常，未找到 base64 或 url")
                else:
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                
                # 图片转换
                img_rgb = img.convert("RGB") 
                img_np = np.array(img_rgb).astype(np.float32) / 255.0 
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                
                return (img_tensor,)
            else:
                raise RuntimeError(f"❌ 未找到图片数据，返回内容: {res_json}")

        except Exception as e:
            print(f"❌ 运行异常: {e}")
            raise e

# --- 注册节点 ---
NODE_CLASS_MAPPINGS = {
    "JM_Seedream_Universal_T2I": Seeddream_Universal_T2I
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JM_Seedream_Universal_T2I": "JM:Seedream Universal T2I (4.0/4.5)"
}