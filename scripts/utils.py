import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def bezier(t, p0, p1, p2, p3):
    t = t.view(-1, 1)
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def bezier_derivative(t, p0, p1, p2, p3):
    t = t.view(-1, 1)
    return 3*(1-t)**2 * (p1 - p0) + 6*(1-t)*t * (p2 - p1) + 3*t**2 * (p3 - p2)

def numpy_bezier(t, p0, p1, p2, p3):
    t = t.reshape(-1, 1)
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def export_standard_svg(filepath, cp_batch, thickness_batch, color_batch, opacity_batch=None, W=512, H=512):
    svg_content = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        f'  <rect width="{W}" height="{H}" fill="white" />'
    ]
    for k in range(len(cp_batch)):
        cp = cp_batch[k]
        d = thickness_batch[k]
        c = color_batch[k]
        o = opacity_batch[k] if opacity_batch is not None else 1.0
        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
        p0 = cp[0] * np.array([W, H]); p1 = cp[1] * np.array([W, H])
        p2 = cp[2] * np.array([W, H]); p3 = cp[3] * np.array([W, H])
        stroke_width = d * W
        path_str = f"M {p0[0]:.2f} {p0[1]:.2f} C {p1[0]:.2f} {p1[1]:.2f}, {p2[0]:.2f} {p2[1]:.2f}, {p3[0]:.2f} {p3[1]:.2f}"
        svg_content.append(
            f'  <path d="{path_str}" '
            f'stroke="rgb({r},{g},{b})" '
            f'stroke-width="{stroke_width:.2f}" '
            f'stroke-opacity="{float(o):.3f}" '
            f'fill="none" '
            f'stroke-linecap="round" '
            f'stroke-linejoin="round" />'
        )
    svg_content.append('</svg>')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg_content))

def save_gvf_visualization(gvf_field, bezier_points, save_path):
    """
    可视化 GVF 梯度场并叠加当前优化好的贝塞尔曲线控制点
    :param gvf_field: (H, W, 2) 的 NumPy 数组，代表每个像素的梯度方向 (u, v)
    :param bezier_points: (N, 2) 归一化在 [0, 1] 范围内的控制点坐标
    :param save_path: 保存路径
    """
    H, W = gvf_field.shape[:2]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    # 提取 U 和 V 方向的梯度
    U = gvf_field[..., 0]
    V = gvf_field[..., 1]
    
    plt.figure(figsize=(8, 8), dpi=300)
    
    # 1. 绘制 GVF 背景流线图
    magnitude = np.sqrt(U**2 + V**2)
    # 使用 streamplot 能够极好地展示“场”的汇聚趋势
    plt.streamplot(X, Y, U, V, color=magnitude, cmap='Blues', density=1.5, linewidth=0.5)
    
    # 2. 映射贝塞尔控制点到绝对图像尺寸
    pts_x = bezier_points[:, 0] * W
    pts_y = bezier_points[:, 1] * H
    
    # 3. 绘制前景贝塞尔曲线点 (使用红色散点)
    plt.scatter(pts_x, pts_y, c='red', s=1, alpha=0.8, label="Control Points")
    
    plt.xlim(0, W)
    plt.ylim(H, 0) # 图像坐标系 Y 轴向下，需要反转
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()