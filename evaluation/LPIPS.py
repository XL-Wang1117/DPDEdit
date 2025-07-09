import lpips
import torch
import numpy as np
from skimage import io
import os

def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        img = np.repeat(img, 3, axis=0)  # (3, H, W) if single-channel
    else:
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
    img = np.expand_dims(img, axis=0)  # (1, C, H, W)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor

def calculate_lpips(clean_folder, noisy_folder):
    # 加载LPIPS模型
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_vgg = lpips.LPIPS(net='vgg')

    # 将模型移动到同一设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_alex.to(device)
    loss_fn_vgg.to(device)

    alex_scores = []
    vgg_scores = []

    for clean_filename in os.listdir(clean_folder):
        if clean_filename.endswith('.jpg'):
            prefix = clean_filename.replace('.jpg', '')
            clean_path = os.path.join(clean_folder, clean_filename)
            noisy_filename = f"{prefix}.jpg"
            noisy_path = os.path.join(noisy_folder, noisy_filename)

            if os.path.isfile(clean_path) and os.path.isfile(noisy_path):
                clean = io.imread(clean_path)
                noisy = io.imread(noisy_path)

                # 将图片转换为张量
                clean_tensor = img2tensor(clean).to(device)
                noisy_tensor = img2tensor(noisy).to(device)

                # 计算LPIPS
                lpips_alex = loss_fn_alex(clean_tensor, noisy_tensor).item()
                lpips_vgg = loss_fn_vgg(clean_tensor, noisy_tensor).item()

                alex_scores.append(lpips_alex)
                vgg_scores.append(lpips_vgg)

    avg_lpips_alex = np.mean(alex_scores) if alex_scores else 0
    avg_lpips_vgg = np.mean(vgg_scores) if vgg_scores else 0

    return avg_lpips_alex, avg_lpips_vgg

clean_folder = '/home/wxl/code/Fashion-edit/result/models/'
noisy_folder = '/home/wxl/code/Fashion-edit/result/sdxl_result'

avg_lpips_alex, avg_lpips_vgg = calculate_lpips(clean_folder, noisy_folder)

print("平均LPIPS (AlexNet):", avg_lpips_alex)
print("平均LPIPS (VGG):", avg_lpips_vgg)
