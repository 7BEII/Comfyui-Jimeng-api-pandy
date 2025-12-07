import os
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import requests
import urllib3
import math

# 禁用 SSL 安全警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Seeddream_45_MultiInput_Auto:
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
                # --- 修改：使用 size 下拉菜单 (含 Auto) ---
                "size": (["Auto", "3:4", "1:1", "4:3", "2:3", "3:2", "16:9", "9:16"], {
                    "default": "Auto"
                }),
                # -------------------------------------------------------
                "watermark": ("BOOLEAN", {"default": False}),
            },
            # --- 多图可选输入 ---
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

    def generate_image(self, api_key, endpoint_id, prompt, size, watermark, 
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

        # --- 核心逻辑修改：Auto 模式计算 ---
        target_min_pixels = 3686400 # 1920 * 1920 基准
        
        if size == "Auto":
            # 使用第一张图作为尺寸参考
            ref_image = input_images[0]
            # ComfyUI image shape: [Batch, Height, Width, Channels]
            input_h = ref_image.shape[1]
            input_w = ref_image.shape[2]
            aspect_ratio = input_w / input_h
            
            print(f"🔍 [Auto模式] 参考图1尺寸: {input_w}x{input_h} (比例: {aspect_ratio:.2f})")
            
            # 计算目标宽高：Area = W * H = (H * AR) * H = H^2 * AR
            # => H = sqrt(Area / AR)
            new_h = math.sqrt(target_min_pixels / aspect_ratio)
            new_w = new_h * aspect_ratio
            
            # 对齐到 64 的倍数
            width = int(round(new_w / 64) * 64)
            height = int(round(new_h / 64) * 64)
            
            # 再次检查像素总量，如果不够则补足
            if width * height < target_min_pixels:
                width += 64
                height += 64
                
        else:
            # 固定尺寸映射
            size_map = {
                "1:1":  (2048, 2048),
                "3:4":  (1728, 2304),
                "4:3":  (2304, 1728),
                "2:3":  (1600, 2400),
                "3:2":  (2400, 1600),
                "16:9": (2560, 1440), 
                "9:16": (1440, 2560)
            }
            width, height = size_map.get(size, (2048, 2048))

        # -------------------------------------------------------

        # 2. 像素总量最终检查
        total_pixels = width * height
        if total_pixels < target_min_pixels:
            print(f"⚠️ 警告: 计算后的分辨率 {width}x{height}={total_pixels} 仍略小于建议值 {target_min_pixels}。")

        # 3. 批量转换 Base64
        image_list_base64 = []
        for img in input_images:
            b64_str = self._single_tensor_to_base64(img)
            image_list_base64.append(b64_str)

        # 4. 基础校验
        if not api_key:
            api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("❌ 错误：API Key 不能为空！")
        if not endpoint_id.startswith("ep-"):
            raise ValueError(f"❌ 参数错误：Endpoint ID 必须是 'ep-' 开头。")

        # 5. 构造请求
        size_str = f"{width}x{height}"
        print(f"📏 最终生成尺寸: {size_str} (模式: {size}, 总像素: {total_pixels})")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": endpoint_id,
            "prompt": prompt,
            "image": image_list_base64, # Base64 字符串列表
            "sequential_image_generation": "disabled",
            "response_format": "b64_json",
            "size": size_str,
            "stream": False,
            "watermark": watermark
        }

        print(f"🚀 发送请求到: {endpoint_id}...")

        # 6. 发送请求
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
    "JM_Seeddream_45_MultiImage_Auto": Seeddream_45_MultiInput_Auto
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JM_Seeddream_45_MultiImage_Auto": "JM:Seeddream 4.5 multi_image (Auto Size)"
}