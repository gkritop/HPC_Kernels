import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ci95(x):
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return (0.0, 0.0)
    
    lo, hi = np.quantile(x, [0.025, 0.975])
    med = np.median(x)

    return (med - lo, hi - med)

def load_csvs(paths):
    frames = []

    for p in paths:
        df = pd.read_csv(p)

        need = ["timestamp","op","M","N","K","size","dtype","reps","ns_per_rep","gflops","gbps","checksum"]

        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"[error] {p} missing columns: {missing}")
        
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def summarize(df):
    df = df.copy()

    fallback = (df["size"] <= 0) & (df["M"] > 0)
    df.loc[fallback, "size"] = (df.loc[fallback, "M"] * df.loc[fallback, "N"] * df.loc[fallback, "K"]).astype("int64")

    rows = []
    for (op, size, dtype), g in df.groupby(["op","size","dtype"]):
        gflops_med = np.median(g["gflops"])
        gflops_lo, gflops_hi = ci95(g["gflops"])

        gbps_med = np.median(g["gbps"])
        gbps_lo, gbps_hi = ci95(g["gbps"])

        t_med = np.median(g["ns_per_rep"]) * 1e-9
        t_lo, t_hi = ci95(g["ns_per_rep"] * 1e-9)

        rows.append(dict(op=op,size=int(size),dtype=dtype, gflops=gflops_med, gflops_lo=gflops_lo, gflops_hi=gflops_hi,
                         gbps=gbps_med, gbps_lo=gbps_lo, gbps_hi=gbps_hi, t=t_med, t_lo=t_lo, t_hi=t_hi))
        
    out = pd.DataFrame(rows).sort_values(["op","dtype","size"])
    return out


def style(ax, title, xlabel, ylabel, logx=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logx:
        try:
            ax.set_xscale("log", base=2)
        except Exception:
            pass

    ax.grid(True, which="both", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend(frameon=False)
    else:
        # Make it obvious the plot is empty rather than silently blank
        ax.text(0.5, 0.5, "No matching data to plot", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, alpha=0.7)

def plot_metric(df, metric, ylabel, outpath_prefix):
    for op, g in df.groupby("op"):
        fig, ax = plt.subplots(figsize=(7,4.5))
        
        for dtype, gg in g.groupby("dtype"):
            gg = gg.sort_values("size")
            y = gg[metric].to_numpy()

            lo = gg.get(f"{metric}_lo", pd.Series(np.zeros(len(gg)))).to_numpy()
            hi = gg.get(f"{metric}_hi", pd.Series(np.zeros(len(gg)))).to_numpy()

            yerr = np.vstack((lo, hi))
            ax.errorbar(gg["size"], y, yerr=yerr, fmt="o-", ms=4, lw=1.4, capsize=3, label=f"{dtype}")

        style(ax, f"{op} — {ylabel}", "Problem size (elements or M·N·K)", ylabel)
        fig.tight_layout()
        out_png = f"{outpath_prefix}_{op}.png"
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(out_png, dpi=150)
        plt.close(fig)

def plot_speedup(df, baseline_op, outdir):
    base = df[df["op"]==baseline_op][["size","dtype","t"]].rename(columns={"t":"t_base"})
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7,4.5))
    plotted_any = False

    if base.empty:
        print(f"[warn] no rows for baseline '{baseline_op}'")

    # Draw baseline line(s) at speedup == 1 for each dtype present
    for dtype, bb in base.groupby("dtype"):
        bb = bb.sort_values("size")

        if not bb.empty:
            ax.plot(bb["size"], np.ones(len(bb)), "--", lw=1.2, label=f"{baseline_op} ({dtype})")
            plotted_any = True

    # Plot comparisons that match baseline on size & dtype
    for op, g in df.groupby("op"):
        if op == baseline_op:
            continue

        for dtype, gg in g.groupby("dtype"):
            merged = pd.merge(gg[["size","dtype","t"]], base[base["dtype"]==dtype], on=["size","dtype"], how="inner")
            if merged.empty:
                continue

            spd = merged["t_base"] / merged["t"]
            ax.plot(merged["size"], spd, "o-", ms=4, lw=1.4, label=f"{op} ({dtype})")
            plotted_any = True

    style(ax, f"Speedup vs {baseline_op}", "Problem size", "Speedup (×)")
    fig.tight_layout()

    plt.savefig(outdir / f"speedup_vs_{baseline_op}.png", dpi=150)
    plt.close(fig)


def plot_roofline(df, outdir, peak_flops, peak_bw):
    fig, ax = plt.subplots(figsize=(7,4.5))
    intensities = np.logspace(-3, 3, 512)
    roof = np.minimum(peak_flops, peak_bw * intensities)
    ax.plot(intensities, roof, "-", lw=2, label=f"Roofline ({peak_flops:.1f} GF/s, {peak_bw:.1f} GB/s)")

    for op, g in df.groupby("op"):
        for dtype, gg in g.groupby("dtype"):
            inten = (gg["gflops"]/gg["gbps"]).replace([np.inf, -np.inf], np.nan).dropna()
            perf = gg.loc[inten.index, "gflops"]
            ax.scatter(inten, perf, s=24, label=f"{op} ({dtype})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (F/B)")
    ax.set_ylabel("Attained performance (GFLOP/s)")
    ax.grid(True, which="both", alpha=0.25)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, ncols=2)

    fig.tight_layout()
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    plt.savefig(outdir / "roofline.png", dpi=150)
    plt.close(fig)

def main():
    # TODO: maybe add PDF export for reports later
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="benchmark CSV files")
    ap.add_argument("--outdir", default="plots", help="output directory for figures")
    ap.add_argument("--baseline", default="", help="baseline op for speedup (e.g., 'matmul_naive')")
    ap.add_argument("--roofline", default="", help="GFLOPS:GBPS, e.g., 220:60")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_csvs(args.csvs)
    summary = summarize(df)

    plot_metric(summary, "gflops", "GFLOP/s (median, 95% CI)", f"{outdir}/gflops")
    plot_metric(summary, "gbps", "GB/s (median, 95% CI)", f"{outdir}/gbps")

    if args.baseline:
        plot_speedup(summary, args.baseline, outdir)

    if args.roofline:
        try:
            g, b = args.roofline.split(":")
            plot_roofline(summary, outdir, float(g), float(b))
        except Exception:
            print("[warn] --roofline expects 'GFLOPS:GBPS' (e.g., 220:60)")

    print(f"[ok] wrote figures to {outdir}/")

if __name__ == "__main__":
    main()