import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class VectorizationDataset(Dataset):
    """
    用于可微矢量化任务的标准图像数据集。
    支持自适应下采样：仅当图像长边超过 max_res 时，才保持比例进行下采样。
    """
    def __init__(self, data_dir, max_res=800):
        """
        :param data_dir: 存放测试图像的文件夹路径
        :param max_res: 限制的最大分辨率（长边像素）。超过此值才会触发等比例下采样。
        """
        super().__init__()
        self.data_dir = data_dir
        self.max_res = max_res
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.lower().endswith(valid_exts)
        ])
        
        if len(self.image_paths) == 0:
            print(f"⚠️ 警告：在 {data_dir} 中未找到任何图像文件！")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # 1. 获取原始图像尺寸
        W_orig, H_orig = img.size
        
        # 2. 判断是否需要下采样
        if max(W_orig, H_orig) > self.max_res:
            # 计算缩放比例，使长边刚好等于 max_res
            scale = self.max_res / float(max(W_orig, H_orig))
            new_W = max(1, int(W_orig * scale))
            new_H = max(1, int(H_orig * scale))
        else:
            # 图像不大，保持原始分辨率
            new_W = W_orig
            new_H = H_orig
        
        # 3. 动态应用双线性下采样与张量转换
        transform = transforms.Compose([
            transforms.Resize((new_H, new_W), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        
        img_tensor = transform(img)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        return img_tensor, img_name, new_H, new_W