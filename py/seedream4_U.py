import os
import torch
import numpy as np
from PIL import Image
import requests
import json
import time
import base64
from io import BytesIO
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import math

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Seedream_Universal4_5:
    """
    Seedream 通用版 (4.0/4.5) - 自动修正分辨率版
    - 修复了 4K 分辨率因对齐导致像素超标的问题
    - 自动将总像素限制在 16777216 (API上限) 以内
    """
    
    MODEL_MAP = {
        "Seedream 4.5 (doubao-seedream-4-5-251128)": "doubao-seedream-4-5-251128",
        "Seedream 4.0 (doubao-seedream-4-0-250828)": "doubao-seedream-4-0-250828"
    }
    
    def __init__(self):
        pass
    
    def create_robust_session(self):
        session = requests.Session()
        session.trust_env = False
        retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.timeout = (60, 600)
        session.headers.update({'Connection': 'keep-alive', 'User-Agent': 'ComfyUI-Seedream-Node/1.0'})
        return session
    
    def tensor_to_base64(self, tensor_image):
        if len(tensor_image.shape) == 4: tensor_image = tensor_image[0]
        image_np = tensor_image.cpu().numpy()
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8) if image_np.dtype in [np.float32, np.float64] else image_np.astype(np.uint8)
        if image_np.shape[-1] != 3: image_np = np.transpose(image_np, (1, 2, 0))
        pil_image = Image.fromarray(image_np)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    
    def get_dimensions(self, aspect_ratio, resolution, ref_image=None):
        """计算目标宽高，包含防超标逻辑"""
        
        # API 严格上限
        MAX_PIXELS = 16777216
        
        pixel_counts = {
            "1K": 1024 * 1024,
            "2K": 2048 * 2048,
            "3K": 3072 * 3072,
            "4K": 4096 * 4096 # 这本身就是上限
        }
        
        target_pixels = pixel_counts.get(resolution, 2048*2048)
        
        # 1. 确定比例
        w_ratio, h_ratio = (3, 4)
        if aspect_ratio == "auto":
            if ref_image is not None:
                try:
                    if len(ref_image.shape) == 4:
                        h_px, w_px = ref_image.shape[1], ref_image.shape[2]
                    else:
                        h_px, w_px = ref_image.shape[0], ref_image.shape[1]
                    w_ratio, h_ratio = w_px, h_px
                    print(f"📏 [Auto] 参考图比例: {w_px}x{h_px} ({w_px/h_px:.2f})")
                except:
                    w_ratio, h_ratio = (3, 4)
            else:
                w_ratio, h_ratio = (3, 4)
        else:
            ratios = {"1:1":(1,1), "3:4":(3,4), "4:3":(4,3), "16:9":(16,9), "9:16":(9,16), "21:9":(21,9)}
            w_ratio, h_ratio = ratios.get(aspect_ratio, (3, 4))

        # 2. 初始计算
        ratio_val = w_ratio / h_ratio
        h = (target_pixels / ratio_val) ** 0.5
        w = h * ratio_val
        
        # 3. 对齐 64 (向上取整可能导致超标)
        w = int(((w + 63) // 64) * 64)
        h = int(((h + 63) // 64) * 64)
        
        # 4. === 安全检查与修正 (关键修复) ===
        # 如果总像素超过限制，循环减少尺寸直到合规
        while w * h > MAX_PIXELS:
            # 优先减少较长边，保持比例观感
            if w > h:
                w -= 64
            else:
                h -= 64
            # 防止死循环（理论上不会，但做个保底）
            if w < 64 or h < 64: break
            
        return w, h

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODEL_MAP.keys()), {"default": "Seedream 4.5 (doubao-seedream-4-5-251128)"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Generate anime style"}),
                "api_key": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "aspect_ratio": (["auto", "1:1", "3:4", "4:3", "16:9", "9:16", "21:9"], {"default": "3:4"}),
                "resolution": (["1K", "2K", "3K", "4K"], {"default": "2K"}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "watermark": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "image1": ("IMAGE", {}), "image2": ("IMAGE", {}), 
                "image3": ("IMAGE", {}), "image4": ("IMAGE", {})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("images", "task_id", "status", "batch_count", "generation_time")
    FUNCTION = "generate_images"
    CATEGORY = "JM/AI生成"

    def generate_images(self, model, prompt, api_key, batch_size, aspect_ratio, resolution, guidance_scale,
                       watermark=False, image1=None, image2=None, image3=None, image4=None):
        
        start_time = time.time()
        
        if not api_key:
            apikey_path = os.path.join(os.path.dirname(__file__), "apikey.txt")
            if os.path.exists(apikey_path):
                with open(apikey_path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
        if not api_key: raise ValueError("请填写 API Key")

        model_id = self.MODEL_MAP.get(model, "doubao-seedream-4-0-250828")
        
        # 确定参考图
        ref_img = next((img for img in [image1, image2, image3, image4] if img is not None), None)
        
        # 计算尺寸
        width, height = self.get_dimensions(aspect_ratio, resolution, ref_image=ref_img)
        
        print(f"🚀 生成参数: {model_id} | {width}x{height} (像素: {width*height}) | 限制: 16777216")

        input_images = [img for img in [image1, image2, image3, image4] if img is not None]
        session = self.create_robust_session()
        api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            data = {
                "model": model_id, "prompt": prompt,
                "size": f"{width}x{height}", "n": batch_size, 
                "response_format": "url", "watermark": watermark
            }
            if guidance_scale != 7.5: data["guidance_scale"] = guidance_scale

            if input_images:
                data["image"] = [self.tensor_to_base64(img) for img in input_images]
                data["sequential_image_generation"] = "disabled"

            response = session.post(api_endpoint, headers=headers, json=data, verify=False)
            
            if response.status_code != 200:
                raise RuntimeError(f"API请求失败 ({response.status_code}): {response.text}")

            result = response.json()
            images = []
            if "data" in result:
                for item in result["data"]:
                    images.append(self.download_image(session, item["url"]))
            
            if not images: raise RuntimeError("API返回成功但无图片")

            final_imgs = torch.stack(images)
            return (final_imgs, result.get("task_id", "N/A"), "completed", len(images), time.time()-start_time)

        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            raise

    def download_image(self, session, url):
        resp = session.get(url, timeout=120, verify=False)
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

NODE_CLASS_MAPPINGS = { "JM:Seedream Universal (4.0/4.5)": Seedream_Universal4_5 }
NODE_DISPLAY_NAME_MAPPINGS = { "JM:Seedream Universal (4.0/4.5)": "JM:Seedream Universal (4.0/4.5)" }