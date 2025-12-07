import os
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import requests
import urllib3

# 禁用 SSL 安全警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Seeddream_45_MultiInput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "sk-xxx (必填)"
                }),
                "endpoint_id": ("STRING", {
                    "default": "ep-20251204151256-bmd5x", 
                    "multiline": False,
                    "placeholder": "必须是 ep- 开头的 ID"
                }),
                "prompt": ("STRING", {
                    "default": "将图1的服装换为图2的服装", 
                    "multiline": True
                }),
                # 默认高清尺寸
                "width": ("INT", {
                    "default": 2048, 
                    "min": 1024, 
                    "max": 4096, 
                    "step": 64, 
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 2048, 
                    "min": 1024, 
                    "max": 4096, 
                    "step": 64, 
                    "display": "number"
                }),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            # --- 关键修改：多图可选输入 ---
            "optional": {
                "image1": ("IMAGE", {"tooltip": "参考图 1"}),
                "image2": ("IMAGE", {"tooltip": "参考图 2"}),
                "image3": ("IMAGE", {"tooltip": "参考图 3"}),
                "image4": ("IMAGE", {"tooltip": "参考图 4"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "✨即梦AI生成"
    
    # 辅助：处理单张 Tensor 转 Base64
    def _single_tensor_to_base64(self, image_tensor):
        # 即使是单张输入，ComfyUI 也是 [1, H, W, C]
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
            
        img_np = image_tensor.cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        # 添加头部
        return f"data:image/png;base64,{img_b64}"

    def generate_image(self, api_key, endpoint_id, prompt, width, height, watermark, 
                       image1=None, image2=None, image3=None, image4=None):
        
        # 1. 收集所有输入的图片
        input_images = []
        if image1 is not None: input_images.append(image1)
        if image2 is not None: input_images.append(image2)
        if image3 is not None: input_images.append(image3)
        if image4 is not None: input_images.append(image4)

        if not input_images:
            raise ValueError("❌ 错误：至少需要连接 1 张参考图片 (image1 ~ image4)！")

        print(f"📸 [Multi-Image] 检测到 {len(input_images)} 张参考图输入")

        # 2. 批量转换 Base64
        image_list_base64 = []
        for img in input_images:
            # 这里的 img 是 ComfyUI 的 Tensor
            b64_str = self._single_tensor_to_base64(img)
            image_list_base64.append(b64_str)

        # 3. 基础校验 & 像素检查
        total_pixels = width * height
        min_pixels = 3686400
        if total_pixels < min_pixels:
             print(f"⚠️ 警告: 当前分辨率 {width}x{height} 可能小于模型要求的最小值。")

        if not api_key:
            api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("❌ 错误：API Key 不能为空！")
        if not endpoint_id.startswith("ep-"):
            raise ValueError(f"❌ 参数错误：Endpoint ID 必须是 'ep-' 开头。")

        # 4. 构造请求
        size_str = f"{width}x{height}"
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": endpoint_id,
            "prompt": prompt,
            "image": image_list_base64, # 这里是 Base64 字符串列表
            "sequential_image_generation": "disabled",
            "response_format": "b64_json",
            "size": size_str,
            "stream": False,
            "watermark": watermark
        }

        print(f"🚀 发送请求到: {endpoint_id}...")

        # 5. 发送请求 (抗干扰)
        try:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=3)
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            response = session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=180,
                verify=False,
                proxies={"http": None, "https": None}
            )
            
            if response.status_code != 200:
                error_msg = f"❌ API 请求失败 (状态码 {response.status_code}):\n{response.text}"
                print(error_msg)
                raise RuntimeError(error_msg)

            res_json = response.json()
            
            if "data" in res_json and len(res_json["data"]) > 0:
                b64_data = res_json["data"][0].get("b64_json")
                if not b64_data:
                     raise RuntimeError("API 未返回 Base64 数据")
                
                img_out = Image.open(BytesIO(base64.b64decode(b64_data)))
                img_rgb = img_out.convert("RGB")
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
    "JM_Seeddream_45_MultiImage_V2": Seeddream_45_MultiInput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JM_Seeddream_45_MultiImage_V2": "JM:Seeddream 4.5 multi_image"
}