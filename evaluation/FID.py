import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = '/home/wxl/code/Fashion-edit/result/models'
generated_images_folder = '/home/wxl/code/Fashion-edit/result/sdxl_result'

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                 batch_size=50,
                                                 device=device,
                                                 dims=2048)
print('FID value:', fid_value)
