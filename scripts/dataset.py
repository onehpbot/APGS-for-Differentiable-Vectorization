import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class VectorizationDataset(Dataset):
    """
    用于可微矢量化任务的标准图像数据集。
    支持自适应下采样：仅当图像长边超过 max_res 时，才保持比例进行下采样。
    支持递归：data_dir 下的多层子文件夹都会被扫描（适配"dataset/类别/图片"结构）。
    """
    def __init__(self, data_dir, max_res=800):
        """
        :param data_dir: 存放测试图像的文件夹路径（递归扫描所有子目录）
        :param max_res: 限制的最大分辨率（长边像素）。超过此值才会触发等比例下采样。
        """
        super().__init__()
        self.data_dir = data_dir
        self.max_res = max_res

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        # 递归收集所有子目录下的图像（大小写不敏感、只保留文件）
        all_paths = glob.glob(os.path.join(data_dir, '**', '*'), recursive=True)
        self.image_paths = sorted(
            [p for p in all_paths if p.lower().endswith(valid_exts) and os.path.isfile(p)]
        )

        if len(self.image_paths) == 0:
            print(f"⚠️ 警告：在 {data_dir}（含子目录）中未找到任何图像文件！")

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
        # 用相对路径做唯一名（多子文件夹防冲突）：cat1/img.png -> cat1_img；直接在 data_dir 下则仍是原名
        rel = os.path.relpath(img_path, self.data_dir)
        img_name = os.path.splitext(rel)[0].replace(os.sep, '_')

        return img_tensor, img_name, new_H, new_W