"""
GVF 引导效果可视化（with/without 对比，非拟合 RGB 图）。
把两组实验拟合到的曲线"中心线"叠到目标边缘图上：
  - w/ GVF：曲线应贴边（中心线沿亮边缘脊线走）。
  - w/o GVF：曲线更可能横穿/漂离边缘。
并量化"曲线采样点到最近 Canny 边缘的距离"分布（越低=越贴边）。

用法：
  python scripts/visualize_gvf_guidance.py \
      --target dataset_local/1.image.png \
      --wo_dir output/viz_wo_gvf --w_dir output/viz_w_gvf \
      --img_name 1.image
"""
import os, re, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_svg_centerlines(svg_path, n_samples=40, min_opacity=0.15, min_width_px=1.0):
    txt = open(svg_path, encoding='utf-8').read()
    W = int(re.search(r'width="(\d+)"', txt).group(1))
    H = int(re.search(r'height="(\d+)"', txt).group(1))
    pat = re.compile(
        r'<path d="M ([\-\d.]+) ([\-\d.]+) C ([\-\d.]+) ([\-\d.]+), ([\-\d.]+) ([\-\d.]+), ([\-\d.]+) ([\-\d.]+)"'
        r'[^>]*stroke-width="([\d.]+)"[^>]*stroke-opacity="([\d.]+)"')
    lines = []
    for m in pat.finditer(txt):
        nums = list(map(float, m.groups()))
        p0, p1, p2, p3 = np.array(nums[0:2]), np.array(nums[2:4]), np.array(nums[4:6]), np.array(nums[6:8])
        width, opacity = nums[8], nums[9]
        if opacity < min_opacity or width < min_width_px:
            continue
        t = np.linspace(0, 1, n_samples).reshape(-1, 1)
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        lines.append(pts)
    return lines, W, H


def edge_distance_map(target_np):
    """每像素到最近 Canny 边缘的距离(px)；返回(dist[H,W], edges[H,W])"""
    gray = cv2.cvtColor((target_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    inv = (edges == 0).astype(np.uint8) * 255  # 边缘=0(目标)，其余=255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist, edges


def sample_distances(lines, dist, W, H):
    """所有中心线采样点到最近边缘的距离（px）"""
    ds = []
    for pts in lines:
        xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
        ds.append(dist[ys, xs])
    return np.concatenate(ds) if ds else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--wo_dir', required=True)
    ap.add_argument('--w_dir', required=True)
    ap.add_argument('--img_name', default='1.image')
    ap.add_argument('--save', default='output/_gvf_viz/guidance_compare.png')
    args = ap.parse_args()

    wo_svg = os.path.join(args.wo_dir, f"{args.img_name}_vector.svg")
    w_svg = os.path.join(args.w_dir, f"{args.img_name}_vector.svg")
    wo_lines, W, H = parse_svg_centerlines(wo_svg)
    w_lines, _, _ = parse_svg_centerlines(w_svg)
    print(f"w/o GVF 有效笔画: {len(wo_lines)} | w/ GVF 有效笔画: {len(w_lines)} (按 opacity>0.15,width>1px 过滤)")

    # 目标图 → 同分辨率
    target = np.array(Image.open(args.target).convert('RGB').resize((W, H), Image.BILINEAR)) / 255.0
    dist, edges = edge_distance_map(target)
    wo_d = sample_distances(wo_lines, dist, W, H)
    w_d = sample_distances(w_lines, dist, W, H)
    print(f"曲线点到最近边缘距离(px):  w/o GVF mean={wo_d.mean():.3f} median={np.median(wo_d):.3f}")
    print(f"                          w/ GVF mean={w_d.mean():.3f} median={np.median(w_d):.3f}")

    # 背景：目标图去色 + 边缘增强（边缘用亮色）
    gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, axis=-1).astype(float) * 0.45  # 压暗
    bg[edges > 0] = np.array([255, 230, 120])  # 边缘高亮黄

    def draw_curves(ax, lines, title):
        ax.imshow(bg.astype(np.uint8))
        for pts in lines:
            ax.plot(pts[:, 0], pts[:, 1], color='#1f77b4', linewidth=0.5, alpha=0.55)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    fig = plt.figure(figsize=(18, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
    ax0 = fig.add_subplot(gs[0]); draw_curves(ax0, wo_lines, '(a) w/o GVF  曲线中心线')
    ax1 = fig.add_subplot(gs[1]); draw_curves(ax1, w_lines, '(b) w/ GVF  曲线中心线')
    ax2 = fig.add_subplot(gs[2])
    bins = np.linspace(0, max(wo_d.max(), w_d.max()) + 0.5, 50)
    ax2.hist(wo_d, bins=bins, alpha=0.55, label=f'w/o GVF (mean {wo_d.mean():.2f}px)', color='#ff7f0e')
    ax2.hist(w_d, bins=bins, alpha=0.55, label=f'w/ GVF (mean {w_d.mean():.2f}px)', color='#1f77b4')
    ax2.set_xlabel('曲线采样点到最近边缘距离 (px)'); ax2.set_ylabel('采样点数')
    ax2.set_title('(c) 边缘对齐分布（越靠左越贴边）', fontsize=13)
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save, dpi=160, bbox_inches='tight')
    print(f"已保存: {args.save}")


if __name__ == '__main__':
    main()
