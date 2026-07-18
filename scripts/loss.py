import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    """生成 1D 高斯核"""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """生成用于 SSIM 计算的 2D 高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """底层 SSIM 计算逻辑"""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class ImageReconstructionLoss(nn.Module):
    def __init__(self, l1_weight=0.0, l2_weight=0.0, ssim_weight=0.0, window_size=11):
        """
        组合损失函数接口：
        通过控制权重的比例，可以自由组合 L1, L2, SSIM Loss。
        """
        super(ImageReconstructionLoss, self).__init__()
        self.l1_w = l1_weight
        self.l2_w = l2_weight
        self.ssim_w = ssim_weight
        
        self.window_size = window_size
        self.channel = 3
        # 预先生成高斯窗口并缓存，避免每次前向传播重复计算
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, pred, target):
        """
        pred, target: 支持 (H, W, C) 或 (B, C, H, W) 格式的张量
        """
        total_loss = 0.0
        
        # 1. 计算 L1 和 L2 Loss (对维度形状不敏感，直接计算)
        if self.l1_w > 0:
            total_loss += self.l1_w * F.l1_loss(pred, target)
            
        if self.l2_w > 0:
            total_loss += self.l2_w * F.mse_loss(pred, target)
            
        # 2. 计算 SSIM Loss
        if self.ssim_w > 0:
            # 自动适配 (H, W, 3) 形状，转换为 SSIM 需要的 (1, 3, H, W) 形状
            if pred.dim() == 3 and pred.size(-1) == 3:
                pred_ssim = pred.unsqueeze(0).permute(0, 3, 1, 2)
                target_ssim = target.unsqueeze(0).permute(0, 3, 1, 2)
            else:
                pred_ssim = pred
                target_ssim = target
                
            # SSIM 的值域在 [0, 1] 之间，越接近 1 越好，因此 Loss 为 (1 - SSIM)
            ssim_val = _ssim(pred_ssim, target_ssim, self.window, self.window_size, self.channel)
            total_loss += self.ssim_w * (1.0 - ssim_val)
            
        return total_loss