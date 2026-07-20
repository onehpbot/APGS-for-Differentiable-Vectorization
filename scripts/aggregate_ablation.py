"""
通用多条件消融聚合（N≥2）。支持两组指标：
  - 重建 RECON = [PSNR, SSIM, LPIPS]   （CUDA 渲染 vs 原图）—— 验证 σ_start 不能太大
  - 一致性 CONS = [cPSNR, cSSIM]        （CUDA 渲染 vs SVG 栅格化）—— 验证 σ_end 要够小
CONS 列缺失（旧输出）时自动跳过。

用法（默认路径，直接对应各 runner 输出，无需手敲 --cond）：
  python scripts/aggregate_ablation.py                 # σ 消融（preset=sigma，ref=default）
  python scripts/aggregate_ablation.py --preset gvf     # GVF 消融（ref=GVF_wo）
缺的目录自动跳过（消融没跑完也能跑已完成的）。也可手动覆盖：
  python scripts/aggregate_ablation.py --cond a=output/a --cond b=output/b --ref a
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RECON = ["PSNR", "SSIM", "LPIPS"]      # CUDA vs GT
CONS = ["cPSNR", "cSSIM"]              # 方案A: CUDA vs SVG
SVG_GT = ["sPSNR", "sSSIM"]            # 方案B: SVG vs GT（与 PSNR 对照看差距）
LOWER_BETTER = {"LPIPS"}


def load_final(d):
    df = pd.read_csv(os.path.join(d, "metrics_report.csv"))
    return df[df["Image"] != "AVERAGE"].copy().set_index("Image")


def load_traces(d):
    return {os.path.basename(f).replace("_metrics_history.json", ""): json.load(open(f))
            for f in glob.glob(os.path.join(d, "*_metrics_history.json"))}


def sem(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def best_name(qt, vals, m):
    """该指标最优条件名"""
    col = f"{m}_mean"
    return vals[col].idxmax() if m not in LOWER_BETTER else vals[col].idxmin()


def main():
    # 预设：对应各消融 runner 的输出目录，不给 --cond 时直接用。
    PRESETS = {
        "sigma": {  # run_sigma_ablation.sh 的输出：output/sigma_<NAME>
            "names": ["start_0p1", "start_0p01", "start_0p005", "start_0p001", "end_0p001", "end_0p0005"],
            "start_axis": ["start_0p1", "start_0p01", "start_0p005", "start_0p001"],  # 固定 end=0.0001，看 收敛+耗时
            "end_axis":   ["end_0p001", "end_0p0005", "start_0p01"],                  # 固定 ratio=100，看 方案A/方案B（start_0p01=0.01→0.0001）
            "dir_fmt": "output/sigma_{name}",
            "ref": "start_0p005",   # = default (0.005→0.0001)
            "out": "output/_sigma_ablation",
        },
        "gvf": {    # run_gvf_ablation_full.sh 的输出：output/GVF_wo, output/GVF_w
            "names": ["GVF_wo", "GVF_w"],
            "dir_fmt": "output/{name}",
            "ref": "GVF_wo",
            "out": "output/_gvf_ablation",
        },
    }

    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", default=None,
                    help="name=dir，可重复；不给则用 --preset 的默认条件集")
    ap.add_argument("--preset", default="sigma", choices=list(PRESETS.keys()),
                    help="默认条件集（对应消融脚本输出路径）")
    ap.add_argument("--ref", default=None, help="基线条件名（不给用 preset 默认）")
    ap.add_argument("--out", default=None, help="输出目录（不给用 preset 默认）")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    if args.cond:
        conds = {}
        for c in args.cond:
            n, d = c.split("=", 1); conds[n] = d
    else:
        conds = {n: p["dir_fmt"].format(name=n) for n in p["names"]}
    ref = args.ref or p["ref"]
    out = args.out or p["out"]
    os.makedirs(out, exist_ok=True)

    # 只保留已存在的目录（消融可能没跑完，缺的跳过）
    missing = [n for n, d in conds.items() if not os.path.isdir(d)]
    for n in missing:
        print(f"⚠️ 跳过不存在的目录: {conds[n]}")
    conds = {n: d for n, d in conds.items() if os.path.isdir(d)}
    names = list(conds.keys())
    assert names, "没有任何条件目录存在，先跑消融脚本。"
    if ref not in conds:
        print(f"⚠️ 基线 {ref} 不在现有条件里，改用 {names[0]}")
        ref = names[0]
    args.out = out

    lines = []
    def log(s=""): print(s); lines.append(s)
    log(f"条件: {names}  | 基线: {ref}")

    finals = {n: load_final(conds[n]) for n in names}
    common_img = sorted(set.intersection(*[set(finals[n].index) for n in names]))
    log(f"公共图片数: {len(common_img)}")
    allcols = set.intersection(*[set(finals[n].columns) for n in names])
    recon = [m for m in RECON if m in allcols]
    cons = [m for m in CONS if m in allcols]
    svg_gt = [m for m in SVG_GT if m in allcols]
    metrics = recon + cons + svg_gt
    log(f"指标 -> 重建{recon}  方案A(CUDA-SVG){cons if cons else '(无)'}  方案B(SVG-GT){svg_gt if svg_gt else '(无)'}")

    # ---------- 质量表 ----------
    rows = []
    for n in names:
        row = {"cond": n}
        for m in metrics:
            v = finals[n].loc[common_img, m].values
            row[f"{m}_mean"] = v.mean(); row[f"{m}_sem"] = sem(v)
        rows.append(row)
    qt = pd.DataFrame(rows).set_index("cond")
    qt.to_csv(os.path.join(args.out, "quality_table.csv"), float_format="%.4f")

    log("\n—— 质量 mean±sem ——")
    log(f"  {'cond':<12}" + "".join(f"{m:>16}" for m in metrics))
    for n in names:
        log(f"  {n:<12}" + "".join(f"{qt.loc[n,f'{m}_mean']:.4f}±{qt.loc[n,f'{m}_sem']:.4f}" for m in metrics))

    # paired Δ vs ref
    log(f"\n—— paired Δ vs 基线 {ref} (★=|Δ|>2sem 显著；↑好/↓差) ——")
    for n in names:
        if n == ref: continue
        parts = []
        for m in metrics:
            d = finals[n].loc[common_img, m].values - finals[ref].loc[common_img, m].values
            mm, ss = d.mean(), sem(d)
            better = (mm > 0) if m not in LOWER_BETTER else (mm < 0)
            sig = "★" if abs(mm) > 2 * ss else " "
            parts.append(f"{m}:{mm:+.3f}±{ss:.3f}{sig}{'↑' if better else '↓'}")
        log(f"  {n:<12} " + "  ".join(parts))

    # ---------- 收敛（仅重建 PSNR）----------
    traces = {n: load_traces(conds[n]) for n in names}
    img_trace = [im for im in common_img if all(im in traces[n] for n in names)]
    palette = plt.cm.tab10(np.linspace(0, 1, len(names)))
    it_full = np.asarray(traces[ref][img_trace[0]]["iteration"]) if img_trace else np.array([0])
    Lmin = min((len(traces[n][img_trace[0]]["psnr"]) for n in names), default=0) if img_trace else 0
    it = it_full[:Lmin]
    _trap = getattr(np, "trapezoid", np.trapz)
    psnr, auc = {}, {}
    if img_trace:
        for n in names:
            arrs = [np.asarray(traces[n][im]["psnr"], float)[:Lmin] for im in img_trace]
            psnr[n] = np.array(arrs); auc[n] = _trap(psnr[n], it, axis=1).mean()
        log(f"\n—— 收敛 (n={len(img_trace)} 图) ——")
        log(f"  {'cond':<12}{'AUC':>10}{'PSNR@1000':>12}{'PSNR@2000':>12}{'PSNR@终':>10}")
        for n in names:
            i1 = int(np.argmin(np.abs(it - 1000))); i2 = int(np.argmin(np.abs(it - 2000)))
            log(f"  {n:<12}{auc[n]:>10.0f}{psnr[n][:,i1].mean():>12.3f}{psnr[n][:,i2].mean():>12.3f}{psnr[n][:,-1].mean():>10.3f}")
    norm = {n: (psnr[n] / psnr[n][:, -1:]) for n in names} if img_trace else {}

    # 训练耗时 OPT(min)（每图已记入 CSV）
    opt = {}
    if "OPT(min)" in allcols:
        for n in names:
            v = finals[n].loc[common_img, "OPT(min)"].values
            opt[n] = (float(v.mean()), sem(v))
    two_axis = "start_axis" in p  # σ 消融：两轴报告

    # ---------- 画图 ----------
    def _bar(ax, clist, mlist, title):
        x = np.arange(len(mlist)); wdt = 0.8 / max(len(clist), 1)
        for i, n in enumerate(clist):
            ci = names.index(n)
            ax.bar(x + i * wdt - 0.4 + wdt / 2, [qt.loc[n, f"{m}_mean"] for m in mlist],
                   wdt, yerr=[qt.loc[n, f"{m}_sem"] for m in mlist], color=palette[ci], label=n, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(mlist); ax.set_title(title)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    if two_axis:
        start_conds = [c for c in p["start_axis"] if c in conds]
        end_conds = [c for c in p["end_axis"] if c in conds]
        fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
        # (a) σ_start 收敛速度：标注收敛 epoch（最后一次低于 99%终值 之后 = 稳定收敛），去掉 sem 带、统一线宽
        if img_trace:
            for n in start_conds:
                ci = names.index(n); mu = psnr[n].mean(0)
                ax[0, 0].plot(it, mu, color=palette[ci], lw=1.7,
                              label=n + (" (ref)" if n == ref else ""))
                thr = 0.99 * mu[-1]
                below = np.where(mu < thr)[0]
                ei = int(below[-1] + 1) if len(below) else 0
                ax[0, 0].scatter([it[ei]], [mu[ei]], color=palette[ci], s=28, zorder=5)
                ax[0, 0].annotate(f"ep{int(it[ei])}", (it[ei], mu[ei]),
                                  textcoords="offset points", xytext=(5, 5),
                                  fontsize=7, color=palette[ci])
        ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("PSNR (dB)")
        ax[0, 0].set_title("(a) σ_start: convergence speed  (● = settled, 99% final)")
        ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=7)
        # (b) σ_start 训练耗时
        if opt and start_conds:
            xs = np.arange(len(start_conds))
            ax[0, 1].bar(xs, [opt[n][0] for n in start_conds], yerr=[opt[n][1] for n in start_conds],
                         color=[palette[names.index(n)] for n in start_conds], capsize=3)
            ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels(start_conds, rotation=15, ha="right")
            ax[0, 1].set_ylabel("OPT (min)")
        ax[0, 1].set_title("(b) σ_start: training time"); ax[0, 1].grid(alpha=0.3, axis="y")
        # (c) 方案A：CUDA vs SVG 直接误差
        if cons:
            _bar(ax[1, 0], end_conds, cons, "(c) scheme A: CUDA vs SVG error")
        else:
            ax[1, 0].axis("off")
        # (d) 方案B：GS-vs-GT (PSNR) vs SVG-vs-GT (sPSNR) + 差距 Δ
        if "sPSNR" in allcols and "PSNR" in allcols and end_conds:
            xs = np.arange(len(end_conds)); wdt = 0.38
            pvals = [qt.loc[n, "PSNR_mean"] for n in end_conds]
            svals = [qt.loc[n, "sPSNR_mean"] for n in end_conds]
            ax[1, 1].bar(xs - wdt / 2, pvals, wdt, yerr=[qt.loc[n, "PSNR_sem"] for n in end_conds],
                         color="#1f77b4", label="GS vs GT (PSNR)", capsize=2)
            ax[1, 1].bar(xs + wdt / 2, svals, wdt, yerr=[qt.loc[n, "sPSNR_sem"] for n in end_conds],
                         color="#ff7f0e", label="SVG vs GT (sPSNR)", capsize=2)
            ymax = max(max(pvals), max(svals))
            for xi, pv, sv in zip(xs, pvals, svals):
                ax[1, 1].annotate(f"Δ{pv - sv:.2f}", (xi, max(pv, sv) + 0.02 * ymax),
                                  ha="center", fontsize=7)
            ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels(end_conds, rotation=15, ha="right")
            ax[1, 1].set_title("(d) scheme B: GS vs SVG (both vs GT)")
            ax[1, 1].grid(alpha=0.3, axis="y"); ax[1, 1].legend(fontsize=7)
        else:
            ax[1, 1].axis("off")
            ax[1, 1].text(0.5, 0.5, "no sPSNR column\n(re-run main.py to get SVG-vs-GT metric)",
                          ha="center", va="center", transform=ax[1, 1].transAxes, fontsize=9)
        plt.tight_layout()
        png = os.path.join(args.out, "convergence.png"); plt.savefig(png, dpi=160, bbox_inches="tight")
        log(f"\n图已存: {png}")

        # ---------- 两轴结论 ----------
        log("\n—— σ_start 轴（收敛速度 + 训练耗时；论点: 起始不能太大）——")
        if img_trace:
            hdr = f"  {'cond':<14}{'AUC':>10}{'PSNR@1000':>12}{'PSNR@2000':>12}" + (f"{'OPT(min)':>14}" if opt else "")
            log(hdr)
            for n in start_conds:
                i1 = int(np.argmin(np.abs(it - 1000))); i2 = int(np.argmin(np.abs(it - 2000)))
                row = f"  {n:<14}{auc[n]:>10.0f}{psnr[n][:, i1].mean():>12.3f}{psnr[n][:, i2].mean():>12.3f}"
                if opt:
                    row += f"{opt[n][0]:>9.2f}±{opt[n][1]:.2f}"
                log(row)
            fast = max(start_conds, key=lambda n: auc[n])
            cheap = min(start_conds, key=lambda n: opt[n][0]) if opt else None
            log(f"  → 收敛最快(AUC): {fast}" + (f"；训练最省时: {cheap}" if cheap else ""))
        log("\n—— σ_end 轴 ——")
        if cons:
            log("  [方案A] CUDA vs SVG 误差 (cPSNR↑ cSSIM↑):")
            log(f"  {'cond':<14}" + "".join(f"{m:>12}" for m in cons))
            for n in end_conds:
                log(f"  {n:<14}" + "".join(f"{qt.loc[n, f'{m}_mean']:>12.3f}" for m in cons))
        if "sPSNR" in allcols:
            log("  [方案B] GS-vs-GT (PSNR) vs SVG-vs-GT (sPSNR)，Δ=PSNR−sPSNR（越小=SVG 越贴近 CUDA 的拟合）:")
            log(f"  {'cond':<14}{'PSNR':>12}{'sPSNR':>12}{'Δ':>10}")
            for n in end_conds:
                pv = qt.loc[n, "PSNR_mean"]; sv = qt.loc[n, "sPSNR_mean"]
                log(f"  {n:<14}{pv:>12.3f}{sv:>12.3f}{pv - sv:>10.3f}")
        if end_conds:
            best_a = max(end_conds, key=lambda n: qt.loc[n, "cPSNR_mean"]) if cons else None
            best_b = (min(end_conds, key=lambda n: qt.loc[n, "PSNR_mean"] - qt.loc[n, "sPSNR_mean"])
                      if "sPSNR" in allcols else None)
            log("  →" + (f" 方案A 最优(cPSNR): {best_a}" if best_a else "")
                + (f"；方案B 最小Δ: {best_b}" if best_b else ""))

    else:
        n_panels = 2 + (1 if recon else 0) + (1 if cons else 0)
        fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.8))
        if n_panels == 1:
            ax = [ax]
        pi = 0
        if img_trace:
            for i, n in enumerate(names):
                mu = psnr[n].mean(0); se = psnr[n].std(0, ddof=1) / np.sqrt(psnr[n].shape[0])
                lw = 2.4 if n == ref else 1.6
                ax[pi].plot(it, mu, color=palette[i], lw=lw, label=n + (" (ref)" if n == ref else ""))
                ax[pi].fill_between(it, mu - se, mu + se, color=palette[i], alpha=0.13)
            ax[pi].set_xlabel("epoch"); ax[pi].set_ylabel("PSNR (dB)")
            ax[pi].set_title("(a) Reconstruction PSNR (mean±sem)"); ax[pi].grid(alpha=0.3); ax[pi].legend(fontsize=7)
            pi += 1
            for i, n in enumerate(names):
                mu = norm[n].mean(0); se = norm[n].std(0, ddof=1) / np.sqrt(norm[n].shape[0])
                lw = 2.4 if n == ref else 1.6
                ax[pi].plot(it, mu, color=palette[i], lw=lw, label=n)
                ax[pi].fill_between(it, mu - se, mu + se, color=palette[i], alpha=0.13)
            ax[pi].axhline(0.9, color="gray", ls=":", lw=1)
            ax[pi].set_xlabel("epoch"); ax[pi].set_ylabel("fraction of final PSNR")
            ax[pi].set_title("(b) Normalized progress"); ax[pi].grid(alpha=0.3); ax[pi].legend(fontsize=7)
            pi += 1
        if recon:
            _bar(ax[pi], names, recon, "(c) Reconstruction (CUDA vs target)"); pi += 1
        if cons:
            _bar(ax[pi], names, cons, "(d) Consistency (CUDA vs SVG)")
        plt.tight_layout()
        png = os.path.join(args.out, "convergence.png"); plt.savefig(png, dpi=160, bbox_inches="tight")
        log(f"\n图已存: {png}")

        log("\n—— 结论 ——")
        if recon:
            log("  [论点1: σ_start 不能太大 → 重建] 各条件重建最优：")
            for m in recon:
                b = best_name(qt, qt, m); log(f"     {m}: {b}  (基线 {ref} {'✓最优' if b==ref else '✗非最优'})")
        if cons:
            log("  [论点2: σ_end 要够小 → 一致性] 各条件一致性最优：")
            for m in cons:
                b = best_name(qt, qt, m); log(f"     {m}: {b}  (基线 {ref} {'✓最优' if b==ref else '✗非最优'})")
        if img_trace:
            ba = max(auc, key=auc.get); log(f"  收敛最快(AUC): {ba}  (基线 {ref} {'✓' if ba==ref else '✗'})")

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结论已存: {os.path.join(args.out, 'summary.txt')}")


if __name__ == "__main__":
    main()
