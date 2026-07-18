import os
import time
import torch
import torch.optim as optim
from PIL import Image
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import resvg_python  # 导入你的 resvg_python
import json
import matplotlib.pyplot as plt

# 指标计算相关
import lpips
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_metric

# 从其他模块导入
from config import Config as cfg
from dataset import VectorizationDataset
from utils import *
from renderer import CUDABezierRenderer
from model import MultiCurveFitter, prune_by_opacity_threshold, sample_new_curves, apply_resize, make_optimizer
from snake_prior import SnakePrior
from loss import ImageReconstructionLoss

def fit_single_image(target_img, img_name, H, W, lpips_vgg, log_interval=50):
    """
    单张图像优化，并返回 (PSNR, SSIM, LPIPS, OPT_Time_in_mins)
    """
    print(f"\n🚀 开始处理: {img_name} | {W}x{H}")
    start_time = time.time()  
    
    target_img = target_img.to(cfg.DEVICE)
    target_img_permuted = target_img.permute(1, 2, 0)
    
    save_dir = cfg.OUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    # === 初始化指标记录器 ===
    metrics_history = {
        "iteration": [],
        "psnr": [],
        "ssim": [],
        "loss": []
    }

    fitter = MultiCurveFitter(num_curves=cfg.INITIAL_CURVES, w_max=cfg.W_MAX).to(cfg.DEVICE)
    renderer = CUDABezierRenderer(H, W, cfg.DEVICE)
    optimizer = make_optimizer(fitter, lr=cfg.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR_MIN)
    
    force_field = SnakePrior.compute_prior_gradient(target_img, kernel_size=25, sigma=8.0)
    loss_fn = ImageReconstructionLoss(l1_weight=cfg.L1_WEIGHT, l2_weight=cfg.L2_WEIGHT, ssim_weight=cfg.SSIM_WEIGHT).to(cfg.DEVICE)

    pbar = tqdm(range(cfg.NUM_EPOCHS + 1), desc=f"Training [{img_name}]", dynamic_ncols=True, leave=False)

    warmup_epochs = int(cfg.NUM_EPOCHS * 0.9)

    for epoch in pbar:
        optimizer.zero_grad()
        cp, thickness, color, opacity = fitter.get_params()
        progress = epoch / cfg.NUM_EPOCHS
        
        if epoch < warmup_epochs:
            # 线性增加：从 0 平滑过渡到传入的 target snake_w
            current_snake_w = cfg.SNAKE_W * (epoch / warmup_epochs)
        else:
            # 预热结束，保持稳定权重
            current_snake_w = cfg.SNAKE_W
        
        current_sigma = cfg.SIGMA_START * (cfg.SIGMA_END / cfg.SIGMA_START) ** progress
        current_tau_op = cfg.TAU_OP_START * (cfg.TAU_OP_END / cfg.TAU_OP_START) ** progress

        rendered, mu_tensor = renderer(cp, thickness, color, opacity, 0.0, current_sigma)
        render_loss = loss_fn(rendered, target_img_permuted)
        snake_loss = SnakePrior.compute_loss(mu_tensor, force_field)
        wr = fitter.width_reg()
        
        loss = render_loss + current_snake_w * snake_loss + cfg.LAMBDA_W * wr + cfg.LAMBDA_OPL1 * opacity.mean()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(fitter.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # pbar.set_postfix({
        #     "Loss": f"{loss.item():.4f}", 
        #     "Snake": f"{snake_loss.item():.6f}"
        # })
        # === 定期记录收敛指标 ===
        if epoch % log_interval == 0:
            with torch.no_grad():
                pred_t = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                tgt_t = target_img.unsqueeze(0).clamp(0, 1)
                cur_psnr = psnr_metric(pred_t, tgt_t, data_range=1.0).item()
                cur_ssim = ssim_metric(pred_t, tgt_t, data_range=1.0).item()
                
                metrics_history["iteration"].append(epoch)
                metrics_history["psnr"].append(float(cur_psnr))
                metrics_history["ssim"].append(float(cur_ssim))
                metrics_history["loss"].append(float(loss.item()))
                
        if epoch > 0 and epoch % cfg.RELEASE_INTERVAL == 0 and fitter.num_curves < cfg.TARGET_CURVES:
            with torch.no_grad():
                err = (rendered - target_img_permuted).abs().mean(-1)
                to_add = min(cfg.CURVES_PER_RELEASE, cfg.TARGET_CURVES - fitter.num_curves)
                new_cp, nrt, nrc, nro = sample_new_curves(err, target_img_permuted, to_add, cfg.DEVICE)
                apply_resize(fitter, torch.ones(fitter.num_curves, dtype=torch.bool, device=cfg.DEVICE), new_cp, nrt, nrc, nro)
                cur_lr = optimizer.param_groups[0]['lr']
                optimizer = make_optimizer(fitter, lr=cur_lr)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.NUM_EPOCHS - epoch), eta_min=cfg.LR_MIN)
                
        elif (cfg.DENSIFY and epoch > 0 and epoch % cfg.DENSIFY_INTERVAL == 0 and epoch < cfg.NUM_EPOCHS - cfg.STOP_BEFORE):
            with torch.no_grad():
                err = (rendered - target_img_permuted).abs().mean(-1)
            keep, n_pruned = prune_by_opacity_threshold(fitter, current_tau_op)
            if n_pruned > 0:
                new_cp, nrt, nrc, nro = sample_new_curves(err, target_img_permuted, n_pruned, cfg.DEVICE)
                apply_resize(fitter, keep, new_cp, nrt, nrc, nro)
                cur_lr = optimizer.param_groups[0]['lr']
                optimizer = make_optimizer(fitter, lr=cur_lr)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.NUM_EPOCHS - epoch), eta_min=cfg.LR_MIN)

# ================= 导出文件与可视化 =================
    # 保存指标 JSON 文件
    json_path = os.path.join(save_dir, f"{img_name}_metrics_history.json")
    with open(json_path, "w") as f:
        json.dump(metrics_history, f)

    end_time = time.time()
    opt_mins = (end_time - start_time) / 60.0  
    
    with torch.no_grad():
        pred_t = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        tgt_t = target_img.unsqueeze(0).clamp(0, 1)
        val_psnr = psnr_metric(pred_t, tgt_t, data_range=1.0).item()
        val_ssim = ssim_metric(pred_t, tgt_t, data_range=1.0).item()
        
        pred_lpips = pred_t * 2.0 - 1.0
        tgt_lpips = tgt_t * 2.0 - 1.0
        val_lpips = lpips_vgg(pred_lpips, tgt_lpips).item()

    # 获取优化完成的参数并转为 numpy
    final_cp, final_thickness, final_color, final_opacity = [x.detach().cpu().numpy() for x in fitter.get_params()]
    
    # 准备可视化所需的数据格式
    gvf_np = force_field.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    target_img_np = target_img_permuted.detach().cpu().numpy()
    
    # 1. 导出图像边界图 (Ground Truth Edge)
    # save_image_boundary(target_img_np, os.path.join(save_dir, f"{img_name}_boundary.png"))
    
    # 2. 导出梯度热力图 (GVF Magnitude)
    # save_gradient_heatmap(gvf_np, os.path.join(save_dir, f"{img_name}_gvf_heatmap.png"))
    
    # 3. 导出曲线边界覆盖图 (Wireframe over Heatmap)
    # save_curve_boundary_overlay(gvf_np, final_cp, os.path.join(save_dir, f"{img_name}_curve_overlay.png"))
    
    # (保留你原有的散点流线图对比)
    # pts_np = final_cp.reshape(-1, 2) 
    # save_gvf_visualization(gvf_np, pts_np, os.path.join(save_dir, f"{img_name}_gvf_stream.png"))
    
    # 渲染结果与 SVG 导出
    Image.fromarray((rendered.detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(save_dir, f"{img_name}_render.png"))
    
    svg_path = os.path.join(save_dir, f"{img_name}_vector.svg")
    export_standard_svg(svg_path, final_cp, final_thickness, final_color, final_opacity, W, H)
    
    '''
    resvg_png_path = os.path.join(save_dir, f"{img_name}_svg_resvg.png")
    try:
        svg_content = open(svg_path, encoding='utf-8').read()
        Image.open(io.BytesIO(bytes(resvg_python.svg_to_png(svg_content)))).save(resvg_png_path)
    except Exception as e:
        print(f"⚠️ resvg 转换失败: {e}")
    '''

    return val_psnr, val_ssim, val_lpips, opt_mins

def run_experiment():
    print(f"=== 开始实验: {cfg.OUT_DIR} ===")
    dataset = VectorizationDataset(cfg.DATA_DIR, max_res=cfg.MAX_RES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 初始化 LPIPS VGG 模型
    lpips_vgg = lpips.LPIPS(net='vgg').to(cfg.DEVICE)
    
    results = []
    
    for batch_idx, (img_tensor, img_name, H_tensor, W_tensor) in enumerate(dataloader):
        name = img_name[0]
        tensor = img_tensor[0]
        H, W = H_tensor.item(), W_tensor.item()
        
        psnr, ssim, lpips_val, opt = fit_single_image(tensor, name, H, W, lpips_vgg, log_interval=args.log_interval)
        
        results.append({
            "Image": name,
            "SSIM": ssim,
            "PSNR": psnr,
            "LPIPS": lpips_val,
            "OPT(min)": opt
        })
        print(f"[{name}] PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {lpips_val:.4f} | Time: {opt:.2f} min")

    # ================= 写入 CSV 与求均值 =================
    df = pd.DataFrame(results)
    mean_row = df.mean(numeric_only=True).to_frame().T
    mean_row['Image'] = 'AVERAGE'
    df = pd.concat([df, mean_row], ignore_index=True)
    
    csv_path = os.path.join(cfg.OUT_DIR, "metrics_report.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n🎉 实验完成！统计结果已保存至: {csv_path}")
    print(df.tail(1).to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='default_run')
    
    # 基础 Loss 参数 (保留之前的)
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--l2_weight', type=float, default=1.0)
    parser.add_argument('--ssim_weight', type=float, default=1.0)
    
    # Group A: 几何先验参数
    parser.add_argument('--snake_w', type=float, default=0.5, help="GVF损失权重")
    parser.add_argument('--lambda_w', type=float, default=10.0, help="宽度正则化权重")
    
    # Group B: 拓扑优化策略
    parser.add_argument('--enable_densify', type=int, default=1, help="1开启自适应剪枝, 0关闭")
    parser.add_argument('--enable_hierarchical', type=int, default=0, help="1开启层次释放, 0关闭")
    
    # Group C: 曲线数量
    parser.add_argument('--target_curves', type=int, default=1024, help="最终期望的曲线数量")
    parser.add_argument('--log_interval', type=int, default=50, help="记录指标的间隔步数")
    
    args = parser.parse_args()
    
    # 应用基础 Loss 配置
    cfg.L1_WEIGHT = args.l1_weight
    cfg.L2_WEIGHT = args.l2_weight
    cfg.SSIM_WEIGHT = args.ssim_weight
    
    # 应用几何先验配置
    cfg.SNAKE_W = args.snake_w
    cfg.LAMBDA_W = args.lambda_w
    
    # 应用拓扑策略
    cfg.DENSIFY = bool(args.enable_densify)
    if not bool(args.enable_hierarchical):
        # 如果关闭层次释放，初始曲线直接等于目标曲线，且把释放间隔设为极大值使其不触发
        cfg.INITIAL_CURVES = args.target_curves
        cfg.TARGET_CURVES = args.target_curves
        cfg.RELEASE_INTERVAL = 999999 
    else:
        cfg.TARGET_CURVES = args.target_curves
        # 保持配置中原有的 INITIAL_CURVES（例如 128 或 256）作为起点
    
    # 创建输出目录
    cfg.OUT_DIR = os.path.join("output", args.exp_name)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    run_experiment()