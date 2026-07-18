"""收敛曲线对比：w/o vs w/ GVF 的 PSNR/SSIM 随 epoch 变化。"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ap = argparse.ArgumentParser()
ap.add_argument('--wo_dir', required=True)
ap.add_argument('--w_dir', required=True)
ap.add_argument('--img_name', default='1.image')
ap.add_argument('--save', default='output/_gvf_viz/convergence.png')
args = ap.parse_args()

wo = json.load(open(os.path.join(args.wo_dir, f"{args.img_name}_metrics_history.json")))
w  = json.load(open(os.path.join(args.w_dir,  f"{args.img_name}_metrics_history.json")))
it = wo['iteration']

# w/ 是否在每个 epoch 都领先？
lead_psnr = sum(1 for i in range(len(it)) if w['psnr'][i] >= wo['psnr'][i])
print(f"PSNR 领先点数: {lead_psnr}/{len(it)}")
print(f"终值  w/o={wo['psnr'][-1]:.3f}  w/={w['psnr'][-1]:.3f}  Δ={w['psnr'][-1]-wo['psnr'][-1]:+.3f}")
# 到达若干 PSNR 目标的 epoch
for tgt in [26.0, 27.0, 28.0]:
    def first_reach(arr):
        for i, v in enumerate(arr):
            if v >= tgt: return it[i]
        return None
    e_wo, e_w = first_reach(wo['psnr']), first_reach(w['psnr'])
    print(f"  到达 {tgt} dB: w/o={e_wo}  w/={e_w}  加速={((e_wo-e_w)/e_wo*100) if (e_wo and e_w) else 'n/a'}%")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(it, wo['psnr'], 'o-', ms=2.5, lw=1.6, color='#ff7f0e', label='w/o GVF')
ax[0].plot(it, w['psnr'],  's-', ms=2.5, lw=1.6, color='#1f77b4', label='w/ GVF')
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('PSNR (dB)'); ax[0].set_title('PSNR 收敛')
ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].plot(it, wo['ssim'], 'o-', ms=2.5, lw=1.6, color='#ff7f0e', label='w/o GVF')
ax[1].plot(it, w['ssim'],  's-', ms=2.5, lw=1.6, color='#1f77b4', label='w/ GVF')
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('SSIM'); ax[1].set_title('SSIM 收敛')
ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout()
os.makedirs(os.path.dirname(args.save), exist_ok=True)
plt.savefig(args.save, dpi=160, bbox_inches='tight')
print(f"已保存: {args.save}")
