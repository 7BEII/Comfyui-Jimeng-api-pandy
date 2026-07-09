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

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Seedream_Universal5:
    """
    Seedream 通用版 (4.0/4.5/5.0) - 自动修正分辨率版
    - 支持 Seedream 5.0 (doubao-seedream-5.0-lite) 模型
    - 支持 Seedream 4.0 / 4.5 模型
    - 修复了 4K 分辨率因对齐导致像素超标的问题
    - 自动将总像素限制在 16777216 (API上限) 以内
    """
    
    DEFAULT_MODEL_ID = "ep-20260709093223-gx4sn"
    DEFAULT_MODEL_NAME = "Seedream 5.0 Pro (doubao-seedream-5-0-pro-260628)"

    MODEL_MAP = {
        DEFAULT_MODEL_NAME: DEFAULT_MODEL_ID,
        "Seedream 5.0 (doubao-seedream-5.0-lite-260128)": "ep-20260428151640-9dfcd",
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
    
    def get_dimensions(self, aspect_ratio, resolution, max_pixels=16777216, min_pixels=3686400, ref_image=None, exact_sizes=None):
        """计算目标宽高，包含防超标逻辑"""
        if exact_sizes and resolution in exact_sizes and aspect_ratio in exact_sizes[resolution]:
            return exact_sizes[resolution][aspect_ratio]
        
        # API 严格上限
        MAX_PIXELS = max_pixels
        MIN_PIXELS = min_pixels
        
        pixel_counts = {
            "1K": 1024 * 1024,
            "2K": 2048 * 2048,
            "3K": 3072 * 3072,
            "4K": 4096 * 4096 # 这本身就是上限
        }
        
        target_pixels = pixel_counts.get(resolution, 2048*2048)
        
        # 0. API 最少像素限制
        if target_pixels < MIN_PIXELS:
            target_pixels = MIN_PIXELS

        # 1. 直接在此处限制总像素，完美确保比例不被破坏
        if target_pixels > MAX_PIXELS:
            target_pixels = MAX_PIXELS
            
        # 2. 确定比例
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
            ratios = {"1:1":(1,1), "2:3":(2,3), "3:2":(3,2), "3:4":(3,4), "4:3":(4,3), "16:9":(16,9), "9:16":(9,16), "21:9":(21,9)}
            w_ratio, h_ratio = ratios.get(aspect_ratio, (3, 4))

        # 3. 计算精确的 w, h
        ratio_val = w_ratio / h_ratio
        h = (target_pixels / ratio_val) ** 0.5
        w = h * ratio_val
        
        # 4. 对齐 64 (改为向下取整，100% 保证不超过 MAX_PIXELS)
        w = int((w // 64) * 64)
        h = int((h // 64) * 64)
        
        # 保底修正：最小边长不得小于 64
        if w < 64: w = 64
        if h < 64: h = 64
        
        return w, h

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODEL_MAP.keys()), {"default": cls.DEFAULT_MODEL_NAME}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Generate anime style"}),
                "api_key": ("STRING", {"default": ""}),
                "custom_model_id": ("STRING", {"default": "", "placeholder": "可选：填写 ep-xxx，留空使用默认 Pro endpoint"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "16:9", "9:16", "21:9"], {"default": "3:4"}),
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

    def generate_images(self, model, prompt, api_key, custom_model_id, batch_size, aspect_ratio, resolution, guidance_scale,
                       watermark=False, image1=None, image2=None, image3=None, image4=None):
        
        start_time = time.time()
        
        if not api_key:
            apikey_path = os.path.join(os.path.dirname(__file__), "apikey.txt")
            if os.path.exists(apikey_path):
                with open(apikey_path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
        if not api_key:
            api_key = os.environ.get("ARK_API_KEY", "").strip()
        if not api_key: raise ValueError("请填写 API Key")

        model_id = custom_model_id.strip() if custom_model_id and custom_model_id.strip() else self.MODEL_MAP.get(model, self.DEFAULT_MODEL_ID)
        is_sd5 = "ep-202" in model_id or "seedream-5" in model_id
        is_sd5_pro = "ep-" in model_id
        use_pro_exact_sizes = "seedream-5-0-pro" in model_id or model_id == "ep-20260709093223-gx4sn"
        
        # 确定参考图
        ref_img = next((img for img in [image1, image2, image3, image4] if img is not None), None)
        
        # Seedream 5.0 Lite 像素上限为 10404496, 4.0为 16777216
        max_pixels = 10404496 if is_sd5 else 16777216
        
        # 计算尺寸
        exact_sizes = {
            "1K": {
                "1:1": (1488, 1488),
                "2:3": (1216, 1824),
                "3:4": (1216, 1824),
                "3:2": (1824, 1216),
                "4:3": (1824, 1216),
            }
        } if use_pro_exact_sizes else None
        width, height = self.get_dimensions(aspect_ratio, resolution, max_pixels=max_pixels, ref_image=ref_img, exact_sizes=exact_sizes)
        
        print(f"🚀 生成参数: {model_id} | {width}x{height} (像素: {width*height}) | 限制: {max_pixels}")

        input_images = [img for img in [image1, image2, image3, image4] if img is not None]
        session = self.create_robust_session()
        api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            
            data = {
                "model": model_id, "prompt": prompt,
                "n": batch_size, 
                "response_format": "url", "watermark": watermark,
                "stream": False
            }
            
            data["size"] = f"{width}x{height}"
                
            if guidance_scale != 7.5: data["guidance_scale"] = guidance_scale

            if input_images:
                # 官方 5.0 文档里单图传的是字符串，而不是列表
                if is_sd5 and len(input_images) == 1:
                    data["image"] = self.tensor_to_base64(input_images[0])
                else:
                    data["image"] = [self.tensor_to_base64(img) for img in input_images]
                if not is_sd5_pro:
                    data["sequential_image_generation"] = "disabled"

            import concurrent.futures
            import comfy.model_management as mm
            
            # 使用独立线程发送请求，主线程负责监听 ComfyUI 的强制中断指令 (Cancel)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(session.post, api_endpoint, headers=headers, json=data, verify=False)
                
                # 每 0.2 秒轮询一次，如果用户点击了 ComfyUI 的取消按钮，立刻报错中断
                while not future.done():
                    mm.throw_exception_if_processing_interrupted()
                    time.sleep(0.2)
                    
                response = future.result()
            
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_info = response.json().get("error", {})
                    error_text = error_info.get("message", response.text)
                    error_param = str(error_info.get("param", "")).lower()
                    error_code = str(error_info.get("code", "")).lower()
                except Exception:
                    error_param = ""
                    error_code = ""

                looks_like_model_error = (
                    response.status_code in (400, 401, 403, 404)
                    and (
                        "model" in error_param
                        or "model" in error_code
                        or "endpoint" in error_code
                        or "model" in error_text.lower()
                        or "endpoint" in error_text.lower()
                        or "not found" in error_text.lower()
                        or "permission" in error_text.lower()
                        or "unauthorized" in error_text.lower()
                    )
                )
                if looks_like_model_error:
                    raise RuntimeError(
                        f"这个 model id/model key 不可用：{model_id}。请检查是否填了当前账号可用的 ep-xxx endpoint。原始错误: {error_text}"
                    )

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

NODE_CLASS_MAPPINGS = { "JM:Seedream Universal (4.0/4.5/5.0)": Seedream_Universal5 }
NODE_DISPLAY_NAME_MAPPINGS = { "JM:Seedream Universal (4.0/4.5/5.0)": "JM:Seedream Universal (4.0/4.5/5.0)" }



