import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
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

def save_image_boundary(target_img_np, save_path):
    """
    1. 图像边界图: 使用 Canny 边缘检测提取 Ground Truth 结构边缘
    :param target_img_np: (H, W, 3) 范围在 [0, 1] 的原图 Numpy 数组
    """
    # 转换为 8-bit 灰度图供 OpenCV 处理
    img_uint8 = (target_img_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # 使用 Canny 提取硬边缘
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    
    plt.figure(figsize=(8, 8), dpi=300)
    # 反转颜色：让背景变白，边缘变黑，方便在论文中展示
    plt.imshow(255 - edges, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_gradient_heatmap(gvf_field, save_path):
    """
    2. 梯度热力图: 展示 GVF 场弥散到平坦区域的幅值强度
    :param gvf_field: (H, W, 2) 的 NumPy 数组
    """
    U = gvf_field[..., 0]
    V = gvf_field[..., 1]
    magnitude = np.sqrt(U**2 + V**2)
    
    plt.figure(figsize=(8, 8), dpi=300)
    # 使用 magma 热力图配色，能很好地突出梯度强度的平滑过渡
    plt.imshow(magnitude, cmap='magma')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_curve_boundary_overlay(gvf_field, bezier_points_raw, save_path):
    """
    3. 曲线边界图: 将贝塞尔控制点拓扑网络半透明叠加在 GVF 热力图上
    :param bezier_points_raw: 原始形状 (N, 4, 2) 或 (N*4, 2) 的控制点在 [0,1] 的坐标
    """
    H, W = gvf_field.shape[:2]
    U = gvf_field[..., 0]
    V = gvf_field[..., 1]
    magnitude = np.sqrt(U**2 + V**2)
    
    plt.figure(figsize=(8, 8), dpi=300)
    # 背景放宽透明度的热力图
    plt.imshow(magnitude, cmap='magma', alpha=0.5)
    
    # 确保控制点被 reshape 为 (N, 4, 2) 以便按曲线绘制线框
    pts_np = bezier_points_raw.reshape(-1, 4, 2)
    pts_x = pts_np[..., 0] * W
    pts_y = pts_np[..., 1] * H
    
    # 绘制贝塞尔曲线的控制多边形 (Wireframe)
    for i in range(pts_np.shape[0]):
        # 连线展示拓扑结构
        plt.plot(pts_x[i], pts_y[i], color='cyan', linewidth=0.5, alpha=0.7)
        # 控制点作为节点
        plt.scatter(pts_x[i], pts_y[i], color='red', s=0.8, alpha=0.9)
        
    plt.xlim(0, W)
    plt.ylim(H, 0) # 确保坐标系与图像一致
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()