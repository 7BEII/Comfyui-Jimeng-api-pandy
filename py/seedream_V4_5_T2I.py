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

class Seeddream_45_T2I:
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
                    "default": "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车...", 
                    "multiline": True
                }),
                "size": (["1K", "2K", "4K"], {"default": "2K"}),
                "watermark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    # 这里修改了分类名称，方便查找
    CATEGORY = "✨即梦AI生成"

    def generate_image(self, api_key, endpoint_id, prompt, size, watermark):
        # 1. 基础校验
        if not api_key:
            api_key = os.environ.get("ARK_API_KEY")
        
        if not api_key:
            raise ValueError("❌ 错误：API Key 不能为空！")
            
        if not endpoint_id.startswith("ep-"):
            raise ValueError(f"❌ 参数错误：Model ID 必须是 'ep-' 开头的推理接入点 ID。\n您填写的 '{endpoint_id}' 是模型名称。")

        # 2. 准备请求
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 文生图 Payload
        payload = {
            "model": endpoint_id,
            "prompt": prompt,
            "size": size,
            "sequential_image_generation": "disabled",
            "response_format": "b64_json",
            "stream": False,
            "watermark": watermark
        }

        print(f"🚀 [JM:Seeddream 4.5_t2i] 发送请求到: {endpoint_id}...")

        # 3. 发送请求 (抗网络干扰)
        try:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=3)
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            response = session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=120,
                verify=False, # 忽略 SSL 验证
                proxies={"http": None, "https": None} # 绕过系统代理
            )
            
            if response.status_code != 200:
                error_msg = f"❌ API 请求失败 (状态码 {response.status_code}):\n{response.text}"
                print(error_msg)
                raise RuntimeError(error_msg)

            # 4. 解析结果
            res_json = response.json()
            
            if "data" in res_json and len(res_json["data"]) > 0:
                b64_data = res_json["data"][0].get("b64_json")
                if not b64_data:
                     raise RuntimeError("SDK 返回了 URL 模式，未返回 Base64")
                
                # --- 图片转换逻辑 (已修复) ---
                img = Image.open(BytesIO(base64.b64decode(b64_data)))
                img_rgb = img.convert("RGB") # 1. 先转颜色模式
                img_np = np.array(img_rgb).astype(np.float32) / 255.0 # 2. 再转数组
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                
                return (img_tensor,)
            else:
                raise RuntimeError(f"❌ 未找到图片数据，返回内容: {res_json}")

        except Exception as e:
            print(f"❌ 运行异常: {e}")
            raise e

# --- 注册节点 (已修改名称) ---
NODE_CLASS_MAPPINGS = {
    "JM_Seeddream_45_T2I": Seeddream_45_T2I
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JM_Seeddream_45_T2I": "JM:Seeddream 4.5_t2i"
}