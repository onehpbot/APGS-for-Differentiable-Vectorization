import os
import json
import matplotlib.pyplot as plt
import argparse

def plot_compare(gvf_dir, nogvf_dir, img_name, save_dir):
    # 读取两个 JSON 文件
    gvf_json = os.path.join(gvf_dir, f"{img_name}_metrics_history.json")
    nogvf_json = os.path.join(nogvf_dir, f"{img_name}_metrics_history.json")
    
    if not os.path.exists(gvf_json) or not os.path.exists(nogvf_json):
        print(f"找不到 {img_name} 的历史数据，跳过。")
        return

    with open(gvf_json, 'r') as f:
        metrics_gvf = json.load(f)
    with open(nogvf_json, 'r') as f:
        metrics_nogvf = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    iterations = metrics_gvf["iteration"]

    # 1. Loss 曲线
    axes[0].plot(iterations, metrics_gvf["loss"], label='With GVF', color='#1f77b4', linewidth=2)
    axes[0].plot(iterations, metrics_nogvf["loss"], label='Without GVF', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[0].set_title('Loss Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. PSNR 曲线
    axes[1].plot(iterations, metrics_gvf["psnr"], label='With GVF', color='#1f77b4', linewidth=2)
    axes[1].plot(iterations, metrics_nogvf["psnr"], label='Without GVF', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[1].set_title('PSNR (Higher is better)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. SSIM 曲线
    axes[2].plot(iterations, metrics_gvf["ssim"], label='With GVF', color='#1f77b4', linewidth=2)
    axes[2].plot(iterations, metrics_nogvf["ssim"], label='Without GVF', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[2].set_title('SSIM (Higher is better)')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('SSIM')
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'Convergence Comparison: {img_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{img_name}_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 成功生成对比曲线: {img_name}_convergence.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gvf_dir', type=str, required=True, help="带有 GVF 的实验输出文件夹")
    parser.add_argument('--nogvf_dir', type=str, required=True, help="不带 GVF 的实验输出文件夹")
    parser.add_argument('--save_dir', type=str, default='output/comparisons', help="曲线图保存位置")
    args = parser.parse_args()

    # 自动遍历文件夹里的 json，找出共有的图像名字并绘图
    gvf_files = [f for f in os.listdir(args.gvf_dir) if f.endswith('_metrics_history.json')]
    for file in gvf_files:
        img_name = file.replace('_metrics_history.json', '')
        plot_compare(args.gvf_dir, args.nogvf_dir, img_name, args.save_dir)