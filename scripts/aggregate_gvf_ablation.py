"""
GVF 消融聚合分析：对比 w/o vs w/ GVF，输出 (1) 质量提升 (2) 收敛速度 两方面证据。

输入：两个条件的输出目录（各含 metrics_report.csv + 每图 *_metrics_history.json）。
输出（默认 output/_gvf_ablation/）：
  - quality_per_image.csv  每图 PSNR/SSIM/LPIPS 及 paired Δ
  - convergence.png        收敛曲线（原始 PSNR、归一化进度、最终质量柱状）
  - summary.txt            质量/收敛的判定结论

注意：CUDA 非确定性使单图有 ~±1dB 噪声；本脚本用 paired Δ（同图抵消难度差异）+
多图均值来压噪（n 张图 sem ≈ std/√n）。判据 |Δ_mean| > 2·sem 视为显著。
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_final(dir_):
    df = pd.read_csv(os.path.join(dir_, "metrics_report.csv"))
    df = df[df["Image"] != "AVERAGE"].copy()
    df = df.set_index("Image")
    return df


def load_traces(dir_):
    traces = {}
    for f in glob.glob(os.path.join(dir_, "*_metrics_history.json")):
        name = os.path.basename(f).replace("_metrics_history.json", "")
        traces[name] = json.load(open(f))
    return traces


def sem(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wo_dir", required=True, help="w/o GVF 输出目录")
    ap.add_argument("--w_dir", required=True, help="w/ GVF 输出目录")
    ap.add_argument("--out", default="output/_gvf_ablation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # ---------- 1. 质量：paired Δ ----------
    wo = load_final(args.wo_dir)
    w = load_final(args.w_dir)
    common_img = sorted(set(wo.index) & set(w.index))
    log(f"共 {len(common_img)} 张图（两条件都有结果）")
    log(f"  w/o 独有: {len(set(wo.index)-set(w.index))} | w/ 独有: {len(set(w.index)-set(wo.index))}（崩溃/缺失）")

    q = pd.DataFrame({"Image": common_img}).set_index("Image")
    verdict = {}
    for col in ["PSNR", "SSIM", "LPIPS"]:
        q[f"{col}_wo"] = wo.loc[common_img, col].values
        q[f"{col}_w"] = w.loc[common_img, col].values
        d = q[f"{col}_w"] - q[f"{col}_wo"]
        q[f"d{col}"] = d.values
        m, s = d.mean(), sem(d)
        sig = abs(m) > 2 * s
        verdict[col] = (m, s, sig)
        better = "↑好" if (m > 0 and col != "LPIPS") or (m < 0 and col == "LPIPS") else "↓差"
        log(f"  {col:5s}: Δ = {m:+.4f} ± {s:.4f}  (n={len(d)})  {'★显著' if sig else '不显著'}  {better}")
    q.to_csv(os.path.join(args.out, "quality_per_image.csv"), float_format="%.4f")

    log("\n—— 质量结论 ——")
    m, s, sig = verdict["PSNR"]
    log(f"PSNR: GVF {'显著' if sig else '未显著'}改变质量 ({m:+.3f}±{s:.3f} dB)。"
        + ("" if sig else " 效应在 CUDA 噪声范围内，不可判定。"))

    # ---------- 2. 收敛速度 ----------
    wo_t = load_traces(args.wo_dir)
    w_t = load_traces(args.w_dir)
    img_trace = [n for n in common_img if n in wo_t and n in w_t]

    # 对齐到公共 iteration 网格（截到各图最短长度，防中途崩溃）
    def stack(traces, key):
        arrs = []
        minlen = min(len(traces[n][key]) for n in img_trace)
        for n in img_trace:
            arrs.append(np.asarray(traces[n][key][:minlen], float))
        return np.array(arrs), minlen

    wo_p, L = stack(wo_t, "psnr")
    w_p, _ = stack(w_t, "psnr")
    wo_s, _ = stack(wo_t, "ssim")
    w_s, _ = stack(w_t, "ssim")
    it = np.asarray(wo_t[img_trace[0]]["iteration"][:L])

    wo_mean, w_mean = wo_p.mean(0), w_p.mean(0)
    wo_sem, w_sem = wo_p.std(0, ddof=1) / np.sqrt(wo_p.shape[0]), w_p.std(0, ddof=1) / np.sqrt(w_p.shape[0])

    # 归一化进度 = PSNR(epoch)/PSNR_final（每图按自己的终值归一）→ 剥离难度差异看速度
    wo_norm = wo_p / wo_p[:, -1:]
    w_norm = w_p / w_p[:, -1:]
    # AUC（每图，再均）
    _trap = getattr(np, "trapezoid", np.trapz)  # numpy 2.x 用 trapezoid，旧版回退 trapz
    auc_wo = _trap(wo_p, it, axis=1).mean()
    auc_w = _trap(w_p, it, axis=1).mean()
    # 各固定 epoch 的 PSNR
    def psnr_at(arr, epoch):
        idx = int(np.argmin(np.abs(it - epoch)))
        return arr[:, idx].mean()
    log("\n—— 收敛速度 ——")
    log(f"  PSNR @1000ep:  w/o={psnr_at(wo_p,1000):.3f}  w/={psnr_at(w_p,1000):.3f}  Δ={psnr_at(w_p,1000)-psnr_at(wo_p,1000):+.3f}")
    log(f"  PSNR @2000ep:  w/o={psnr_at(wo_p,2000):.3f}  w/={psnr_at(w_p,2000):.3f}  Δ={psnr_at(w_p,2000)-psnr_at(wo_p,2000):+.3f}")
    log(f"  PSNR @终:      w/o={wo_mean[-1]:.3f}  w/={w_mean[-1]:.3f}")
    log(f"  AUC(PSNR·ep):  w/o={auc_wo:.0f}  w/={auc_w:.0f}  Δ={auc_w-auc_wo:+.0f} ({(auc_w-auc_wo)/auc_wo*100:+.2f}%)  {'→ GVF 更快收敛' if auc_w>auc_wo else '→ GVF 不更快'}")
    # 归一化进度：到 90% 终值的 epoch（每图），越小说明越快
    def ep_to_frac(arr, frac=0.9):
        eps = []
        for r in arr:
            tgt = frac * r[-1]
            hit = np.where(r >= tgt)[0]
            eps.append(it[hit[0]] if len(hit) else it[-1])
        return np.mean(eps), sem(eps)
    e_wo, se_wo = ep_to_frac(wo_norm, 0.9)
    e_w, se_w = ep_to_frac(w_norm, 0.9)
    log(f"  到达 90% 终值: w/o={e_wo:.0f}±{se_wo:.0f}ep  w/={e_w:.0f}±{se_w:.0f}ep  Δ={e_w-e_wo:+.0f}ep  {'→ GVF 更快' if e_w<e_wo else '→ GVF 不更快'}")

    # ---------- 按数据集拆分（质量 + 收敛）----------
    def dataset_of(name):
        return str(name).split("_", 1)[0]
    ds_of = {n: dataset_of(n) for n in common_img}
    datasets = sorted(set(ds_of.values()), key=lambda d: -sum(1 for n in common_img if ds_of[n] == d))

    def stars(m, s):
        return "★" if abs(m) > 2 * s else " "

    log("\n—— 按数据集拆分 ——")
    log("  [质量] (mean±sem, ★=|Δ|>2sem)")
    log(f"  {'dataset':<14}{'n':>5}  {'PSNR wo→w':>15}  {'dPSNR':>14}{'%>0':>5}  {'dSSIM':>12}  {'dLPIPS':>12}")
    ds_rows = []
    for ds in datasets:
        idx = [n for n in common_img if ds_of[n] == ds]
        if len(idx) < 2:
            continue
        pw = wo.loc[idx, "PSNR"].values; pwf = w.loc[idx, "PSNR"].values
        dp = pwf - pw
        dsim = w.loc[idx, "SSIM"].values - wo.loc[idx, "SSIM"].values
        dlp = w.loc[idx, "LPIPS"].values - wo.loc[idx, "LPIPS"].values
        mp, sp = dp.mean(), sem(dp); ms, ss = dsim.mean(), sem(dsim); ml, sl = dlp.mean(), sem(dlp)
        ds_rows.append(dict(dataset=ds, n=len(idx), PSNR_wo=float(pw.mean()), PSNR_w=float(pwf.mean()),
                            dPSNR=float(mp), dPSNR_sem=float(sp), pct_better=float((dp > 0).mean() * 100),
                            dSSIM=float(ms), dSSIM_sem=float(ss), dLPIPS=float(ml), dLPIPS_sem=float(sl)))
        log(f"  {ds:<14}{len(idx):>5}  {pw.mean():>6.3f}→{pwf.mean():.3f}  {mp:>+8.3f}±{sp:.3f}{stars(mp, sp)}{(dp > 0).mean() * 100:>4.0f}%  {ms:>+7.4f}{stars(ms, ss)}  {ml:>+8.4f}{stars(ml, sl)}")

    log("\n  [收敛] PSNR@1000/@2000/终 / AUC（各数据集内部 w/o vs w/）")
    log(f"  {'dataset':<14}{'n':>5}  {'@1000 wo→w':>15}  {'@2000 wo→w':>15}  {'终 wo→w':>13}  {'AUCΔ%':>8}")
    for ds in datasets:
        idx = [n for n in img_trace if ds_of.get(n) == ds]
        if len(idx) < 2:
            continue
        L = min(len(wo_t[n]["psnr"]) for n in idx)
        it_ds = np.asarray(wo_t[idx[0]]["iteration"][:L])
        wo_p = np.array([wo_t[n]["psnr"][:L] for n in idx], float)
        w_p = np.array([w_t[n]["psnr"][:L] for n in idx], float)

        def at(arr, e):
            i = int(np.argmin(np.abs(it_ds - e))); return arr[:, i].mean()
        auc_wo = _trap(wo_p, it_ds, axis=1).mean(); auc_w = _trap(w_p, it_ds, axis=1).mean()
        for r in ds_rows:
            if r["dataset"] == ds:
                r.update(p1000_wo=float(at(wo_p, 1000)), p1000_w=float(at(w_p, 1000)),
                         p2000_wo=float(at(wo_p, 2000)), p2000_w=float(at(w_p, 2000)),
                         pfinal_wo=float(wo_p[:, -1].mean()), pfinal_w=float(w_p[:, -1].mean()),
                         auc_wo=float(auc_wo), auc_w=float(auc_w), auc_dP=float((auc_w - auc_wo) / auc_wo * 100))
        log(f"  {ds:<14}{len(idx):>5}  {at(wo_p,1000):>6.3f}→{at(w_p,1000):.3f}  {at(wo_p,2000):>6.3f}→{at(w_p,2000):.3f}  {wo_p[:,-1].mean():>5.3f}→{w_p[:,-1].mean():.3f}  {(auc_w-auc_wo)/auc_wo*100:>+7.2f}%")
    if ds_rows:
        pd.DataFrame(ds_rows).to_csv(os.path.join(args.out, "by_dataset.csv"), index=False, float_format="%.4f")
        log(f"\n  按数据集明细已存: {os.path.join(args.out, 'by_dataset.csv')}")

    # ---------- 3. 画图 ----------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    # (a) 原始 PSNR 曲线
    ax[0].plot(it, wo_mean, color="#ff7f0e", lw=1.8, label="w/o GVF")
    ax[0].fill_between(it, wo_mean - wo_sem, wo_mean + wo_sem, color="#ff7f0e", alpha=0.2)
    ax[0].plot(it, w_mean, color="#1f77b4", lw=1.8, label="w/ GVF")
    ax[0].fill_between(it, w_mean - w_sem, w_mean + w_sem, color="#1f77b4", alpha=0.2)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("PSNR (dB)"); ax[0].set_title("(a) PSNR convergence (mean±sem)")
    ax[0].grid(alpha=0.3); ax[0].legend()
    # (b) 归一化进度
    for arr, c, lab in [(wo_norm, "#ff7f0e", "w/o GVF"), (w_norm, "#1f77b4", "w/ GVF")]:
        mu = arr.mean(0); se = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        ax[1].plot(it, mu, color=c, lw=1.8, label=lab)
        ax[1].fill_between(it, mu - se, mu + se, color=c, alpha=0.2)
    ax[1].axhline(0.9, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("fraction of final PSNR"); ax[1].set_title("(b) normalized progress (speed)")
    ax[1].grid(alpha=0.3); ax[1].legend()
    # (c) 最终质量柱状
    metrics = ["PSNR", "SSIM", "LPIPS"]
    wo_bar = [wo.loc[common_img, c].mean() for c in metrics]
    w_bar = [w.loc[common_img, c].mean() for c in metrics]
    wo_err = [sem(wo.loc[common_img, c].values) for c in metrics]
    w_err = [sem(w.loc[common_img, c].values) for c in metrics]
    x = np.arange(len(metrics)); wdt = 0.35
    ax[2].bar(x - wdt / 2, wo_bar, wdt, yerr=wo_err, color="#ff7f0e", label="w/o GVF", capsize=3)
    ax[2].bar(x + wdt / 2, w_bar, wdt, yerr=w_err, color="#1f77b4", label="w/ GVF", capsize=3)
    ax[2].set_xticks(x); ax[2].set_xticklabels(metrics); ax[2].set_title("(c) final quality (mean±sem)")
    ax[2].grid(alpha=0.3, axis="y"); ax[2].legend()
    plt.tight_layout()
    png = os.path.join(args.out, "convergence.png")
    plt.savefig(png, dpi=160, bbox_inches="tight")
    log(f"\n图已存: {png}")

    # 按数据集：@2000ep（早期）与终值（晚期）w/o vs w/，看"早期是否同速、晚期是否领先"
    if ds_rows:
        fig3, ax3 = plt.subplots(1, 2, figsize=(13, 4.6))
        names_ds = [r["dataset"] for r in ds_rows]
        x = np.arange(len(names_ds)); wdt = 0.38
        ax3[0].bar(x - wdt / 2, [r.get("p2000_wo", np.nan) for r in ds_rows], wdt, color="#ff7f0e", label="w/o GVF")
        ax3[0].bar(x + wdt / 2, [r.get("p2000_w", np.nan) for r in ds_rows], wdt, color="#1f77b4", label="w/ GVF")
        ax3[0].set_xticks(x); ax3[0].set_xticklabels(names_ds, rotation=15, ha="right")
        ax3[0].set_ylabel("PSNR @2000ep"); ax3[0].set_title("early convergence by dataset")
        ax3[0].grid(alpha=0.3, axis="y"); ax3[0].legend(fontsize=8)
        ax3[1].bar(x - wdt / 2, [r["PSNR_wo"] for r in ds_rows], wdt, color="#ff7f0e", label="w/o GVF")
        ax3[1].bar(x + wdt / 2, [r["PSNR_w"] for r in ds_rows], wdt, color="#1f77b4", label="w/ GVF")
        ax3[1].set_xticks(x); ax3[1].set_xticklabels(names_ds, rotation=15, ha="right")
        ax3[1].set_ylabel("final PSNR"); ax3[1].set_title("final quality by dataset")
        ax3[1].grid(alpha=0.3, axis="y"); ax3[1].legend(fontsize=8)
        plt.tight_layout()
        png3 = os.path.join(args.out, "by_dataset.png")
        plt.savefig(png3, dpi=160, bbox_inches="tight")
        log(f"按数据集图已存: {png3}")

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结论已存: {os.path.join(args.out, 'summary.txt')}")


if __name__ == "__main__":
    main()
