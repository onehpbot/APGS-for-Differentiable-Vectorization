import torch
import torch.nn.functional as F
import math

class SnakePrior:
    @staticmethod
    def get_gaussian_kernel(kernel_size=21, sigma=5.0, device='cpu'):
        """生成 2D 高斯核，用于扩大引力场的捕获范围"""
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean)**2.0, dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)

    @staticmethod
    def compute_prior_gradient(target_img_tensor, kernel_size=21, sigma=5.0):
        """
        接口1：计算全图的先验梯度引力场
        target_img_tensor: 目标图像 (3, H, W)
        返回: force_field (1, 2, H, W) 记录了全图每个像素受到的 (x, y) 轴拉力
        """
        device = target_img_tensor.device
        
        # 1. 灰度化提取亮度特征
        if target_img_tensor.dim() == 3 and target_img_tensor.shape[0] == 3:
            gray_img = 0.299 * target_img_tensor[0] + 0.587 * target_img_tensor[1] + 0.114 * target_img_tensor[2]
            gray_img = gray_img.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        else:
            gray_img = target_img_tensor.unsqueeze(0) if target_img_tensor.dim() == 3 else target_img_tensor
            
        # 2. 高斯平滑扩散边缘 (让曲线在远处也能感受到拉力)
        gaussian_k = SnakePrior.get_gaussian_kernel(kernel_size, sigma, device)
        pad = kernel_size // 2
        smoothed_img = F.conv2d(F.pad(gray_img, (pad, pad, pad, pad), mode='replicate'), gaussian_k)
        
        # 3. 提取图像能量场 E = |∇I|^2 (哪里反差大，哪里能量越高)
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=device).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32, device=device).unsqueeze(0) / 8.0
        
        Ix = F.conv2d(F.pad(smoothed_img, (1, 1, 1, 1), mode='replicate'), sobel_x)
        Iy = F.conv2d(F.pad(smoothed_img, (1, 1, 1, 1), mode='replicate'), sobel_y)
        edge_energy = Ix**2 + Iy**2
        
        # 4. 计算引力场 V = ∇E (指引曲线向高能量区域爬坡)
        Ex = F.conv2d(F.pad(edge_energy, (1, 1, 1, 1), mode='replicate'), sobel_x)
        Ey = F.conv2d(F.pad(edge_energy, (1, 1, 1, 1), mode='replicate'), sobel_y)
        
        force_field = torch.cat([Ex, Ey], dim=1) # (1, 2, H, W)
        
        # 软归一化：保留拉力方向，防止梯度爆炸，且极度平坦区的拉力自然衰减为 0
        norm = torch.norm(force_field, dim=1, keepdim=True)
        force_field = force_field / (norm + 1e-5)
        
        return force_field

    @staticmethod
    def compute_loss(mu_tensor, force_field):
        """
        接口2：基于微元坐标计算 Snake 势能 Loss
        mu_tensor: 所有微元的物理中心点集合 (N, 2)
        force_field: 预先计算好的全图引力场 (1, 2, H, W)
        """
        if mu_tensor is None or mu_tensor.shape[0] == 0:
            return torch.tensor(0.0, device=force_field.device)
            
        N = mu_tensor.shape[0]
        
        # 1. 将物理坐标 [0, 1] 映射为 grid_sample 所需的 [-1, 1]
        mu_normalized = mu_tensor * 2.0 - 1.0  
        mu_grid = mu_normalized.view(1, -1, 1, 2) # (1, N, 1, 2)
        
        # 2. 从引力场中“采摘”每个微元当前位置受到的拉力向量
        sampled_forces = F.grid_sample(force_field, mu_grid, align_corners=False) # (1, 2, N, 1)
        sampled_forces = sampled_forces.squeeze().t() # 还原为 (N, 2)
        
        # 3. 核心黑魔法：动能内积 Loss
        # f.detach() 切断引力场的梯度，使其纯粹作为方向指示器。
        # 最小化 (-f * mu) 相当于迫使 mu 向 f 的方向移动。
        loss = -torch.mean(torch.sum(sampled_forces.detach() * mu_tensor, dim=-1))
        
        return loss