from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import numpy as np
from PIL import Image

# 初始化 CLIP score 函数
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def apply_mask(image, mask):
    # 确保mask为二值化
    mask = mask.convert('L')
    mask = np.array(mask) > 128
    # 将图像转换为 numpy 数组
    image_np = np.array(image)
    # 创建白色背景
    white_background = np.ones_like(image_np) * 255
    # 将mask应用于图像，使得只保留服装区域，其他部分用白色填充
    image_np[~mask] = white_background[~mask]
    # 将结果转换回PIL图像
    return Image.fromarray(image_np)

def calculate_clip_score(image, prompt):
    # 将图像转换为 255 范围的 uint8 类型
    image_int = (np.array(image) / 255).astype("uint8")
    # 计算 CLIP score
    clip_score_value = clip_score_fn(torch.from_numpy(image_int).permute(2, 0, 1).unsqueeze(0), [prompt]).detach()
    return round(float(clip_score_value), 4)

def main():
    # 直接输入图片路径
    image_path = "/home/wxl/code/Fashion-edit/result/dilate_mask_result/00126_00.jpg"
    mask_path = "/home/wxl/code/Fashion-edit/result/cloth_mask/00126_00.png"
    
    # 打开并转换图像
    selected_image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # 应用mask
    masked_image = apply_mask(selected_image, mask)
    
    # 保存结果图像
    masked_image.save("masked_image.png")
    
    # 用户输入 prompt
    prompt = "pink short-sleeve t-shirts in white background"
    
    # 计算 CLIP score
    sd_clip_score = calculate_clip_score(masked_image, prompt)
    print(f"CLIP score for {image_path} with prompt '{prompt}': {sd_clip_score}")

if __name__ == "__main__":
    main()
