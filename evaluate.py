"""Zero-shot robustness evaluation: perturb mass / friction / actuator delay at TEST time.

  python evaluate.py --backend mujoco --runs runs --out results

Every trained policy (nominal-trained and DR-trained, all seeds) is evaluated on the
same 100 fixed episodes (targets + initial poses) at each perturbation setting, with
no fine-tuning. Outputs: results/sweeps.csv, results/summary.md and figures.
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from arm_common import (MLPPolicy, make_env, rollout, sample_targets, sample_init,
                        tile_conditions, TRAIN_RANGES, DT_CTRL)

MODES = ["nominal", "dr"]
LABEL = {"nominal": "Nominal-trained (no DR)", "dr": "Domain-randomized"}
COLOR = {"nominal": "#d1495b", "dr": "#2e86ab"}
MASS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
FRIC = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
DELAY = [0, 1, 2, 3, 4, 5, 6, 8]
GRID_MASS = [0.5, 0.75, 1.0, 1.5, 2.0]
GRID_FRIC = [0.25, 0.5, 1.0, 2.0, 4.0]
GRID_DELAY = [0, 1, 2, 3, 4, 6]
N_EP = 100


def load(backend, runs):
    pols = {}
    for mode in MODES:
        files = sorted(glob.glob(os.path.join(runs, f"{backend}_{mode}_seed*.npz")))
        d = [np.load(f) for f in files]
        if d:
            pols[mode] = (np.stack([x["theta"] for x in d]), tuple(int(h) for h in d[0]["hidden"]),
                          [x["curve"] for x in d], [x["curve_cols"] for x in d][0])
    return pols


def evaluate_setting(env, policy, thetas, cond, mass=1.0, fric=1.0, delay=0, record=False):
    targets, q0 = cond
    S, B = len(thetas), len(targets) * len(thetas)
    params = {"mass": np.full(B, mass), "friction": np.full(B, fric),
              "delay": np.full(B, delay, dtype=int)}
    _, tt, tq = tile_conditions(S, {"x": np.zeros(len(targets))}, targets, q0)
    r = rollout(env, policy, thetas, params, tt, tq, record=record)
    K = len(targets)
    err = r["final_err"].reshape(S, K)
    return {"err_cm": err.mean(1) * 100, "succ": r["success"].reshape(S, K).mean(1),
            "ret": r["return"].reshape(S, K).mean(1),
            "dist_hist": r["dist_hist"].reshape(-1, S, K)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = make_env(a.backend)
    pols = load(a.backend, a.runs)
    rng = np.random.default_rng(12345)
    cond = (sample_targets(rng, N_EP), sample_init(rng, N_EP))
    rows, sweeps = [], {}

    def run(name, xs, kw):
        for mode, (th, hid, _, _) in pols.items():
            pol = MLPPolicy(hid)
            res = [evaluate_setting(env, pol, th, cond, **kw(x)) for x in xs]
            for key in ("err_cm", "succ", "ret"):
                sweeps[(name, mode, key)] = np.stack([r[key] for r in res])     # (len(xs), S)
            for x, r in zip(xs, res):
                for s in range(len(th)):
                    rows.append([a.backend, mode, name, x, s, r["err_cm"][s], r["succ"][s], r["ret"][s]])

    run("mass", MASS, lambda x: dict(mass=x))
    run("friction", FRIC, lambda x: dict(fric=x))
    run("delay", DELAY, lambda x: dict(delay=x))
    grids = {}
    for gname, gx, key in (("mass_x_delay", GRID_MASS, "mass"), ("friction_x_delay", GRID_FRIC, "fric")):
        for mode, (th, hid, _, _) in pols.items():
            pol = MLPPolicy(hid)
            succ = np.zeros((len(GRID_DELAY), len(gx)))
            err = np.zeros_like(succ)
            for i, dl in enumerate(GRID_DELAY):
                for j, x in enumerate(gx):
                    r = evaluate_setting(env, pol, th, cond, delay=dl, **{key: x})
                    succ[i, j], err[i, j] = r["succ"].mean(), r["err_cm"].mean()
                    for s in range(len(th)):
                        rows.append([a.backend, mode, gname, f"{key}={x},delay={dl}", s,
                                     r["err_cm"][s], r["succ"][s], r["ret"][s]])
            grids[(gname, mode)] = (succ, err)

    with open(os.path.join(a.out, "sweeps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "policy", "sweep", "value", "seed", "final_err_cm", "success_rate", "return"])
        w.writerows(rows)

    # ---- Figure 1: 1-D sweeps ------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
    for col, (name, xs, xl, log) in enumerate([("mass", MASS, "link mass multiplier", False),
                                               ("friction", FRIC, "joint friction multiplier", True),
                                               ("delay", DELAY, f"actuator delay [control steps of {DT_CTRL*1000:.0f} ms]", False)]):
        lo, hi = TRAIN_RANGES[name]
        for row, (key, yl) in enumerate([("succ", "success rate (final error < 5 cm)"),
                                         ("err_cm", "final position error [cm]")]):
            ax = axes[row, col]
            ax.axvspan(lo, hi, color="#999999", alpha=0.15, lw=0, label="DR training range")
            for mode in pols:
                y = sweeps[(name, mode, key)] * (100 if key == "succ" else 1)
                m, s = y.mean(1), y.std(1)
                ax.plot(xs, m, "-o", color=COLOR[mode], label=LABEL[mode], ms=4)
                ax.fill_between(xs, m - s, m + s, color=COLOR[mode], alpha=0.2, lw=0)
            if log:
                ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if key == "err_cm": ax.set_yscale("log")
            ax.set_ylabel(yl if col == 0 else "")
            ax.grid(alpha=0.3)
            if row == 1: ax.set_xlabel(xl)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Zero-shot degradation under test-time perturbations ({a.backend} backend; mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_sweeps.png"), dpi=150); plt.close(fig)

    # ---- Figure 2: 2-D heatmaps ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), gridspec_kw={"hspace": 0.35})
    for r_, (gname, gx, xl, key) in enumerate([("mass_x_delay", GRID_MASS, "mass multiplier", "mass"),
                                              ("friction_x_delay", GRID_FRIC, "friction multiplier", "friction")]):
        for c_, mode in enumerate(MODES):
            ax = axes[r_, c_]
            succ, _ = grids[(gname, mode)]
            im = ax.imshow(succ * 100, vmin=0, vmax=100, cmap="viridis", origin="lower", aspect="auto")
            ax.set_xticks(range(len(gx))); ax.set_xticklabels(gx)
            ax.set_yticks(range(len(GRID_DELAY))); ax.set_yticklabels(GRID_DELAY)
            for i in range(succ.shape[0]):
                for j in range(succ.shape[1]):
                    ax.text(j, i, f"{succ[i, j]*100:.0f}", ha="center", va="center",
                            color="white" if succ[i, j] < 0.6 else "black", fontsize=9)
            lo, hi = TRAIN_RANGES[key]
            inx = [j for j, x in enumerate(gx) if lo <= x <= hi]
            iny = [i for i, d in enumerate(GRID_DELAY) if TRAIN_RANGES["delay"][0] <= d <= TRAIN_RANGES["delay"][1]]
            if inx and iny:
                ax.add_patch(plt.Rectangle((min(inx) - .5, min(iny) - .5), len(inx), len(iny),
                                           fill=False, ec="white", lw=2, ls="--"))
            ax.set_xlabel(xl); ax.set_ylabel("actuator delay [steps]"); ax.set_title(LABEL[mode])
    fig.colorbar(im, ax=axes, label="success rate [%]", shrink=0.8)
    fig.suptitle("Success rate over joint perturbations (dashed box = DR training range)")
    fig.savefig(os.path.join(a.out, "fig_heatmaps.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 3: error vs time under four conditions ----------------------
    conds = [("nominal", dict()), ("mass x1.75", dict(mass=1.75)),
             ("friction x3", dict(fric=3.0)), ("delay 5 steps", dict(delay=5))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (title, kw) in zip(axes, conds):
        for mode, (th, hid, _, _) in pols.items():
            r = evaluate_setting(env, MLPPolicy(hid), th, cond, **kw)
            d = r["dist_hist"].mean(2) * 100             # (T, S)
            t = np.arange(d.shape[0]) * DT_CTRL
            ax.plot(t, d.mean(1), color=COLOR[mode], label=LABEL[mode])
            ax.fill_between(t, d.min(1), d.max(1), color=COLOR[mode], alpha=0.2, lw=0)
        ax.set_title(title); ax.set_xlabel("time [s]"); ax.grid(alpha=0.3); ax.set_yscale("log")
    axes[0].set_ylabel("distance to target [cm]"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_error_vs_time.png"), dpi=150); plt.close(fig)

    # ---- Figure 4: training curves ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (col, title) in zip(axes, [(3, "evaluated on nominal physics"), (6, "evaluated on randomized physics")]):
        for mode, (_, _, curves, _) in pols.items():
            for k, c in enumerate(curves):
                ax.plot(c[:, 0], c[:, col] * 100, color=COLOR[mode], alpha=0.8,
                        label=LABEL[mode] if k == 0 else None)
        ax.set_yscale("log"); ax.set_xlabel("ES generation"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation final error [cm]"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_training.png"), dpi=150); plt.close(fig)

    # ---- Markdown summary ---------------------------------------------------
    def cell(name, mode, xi):
        e = sweeps[(name, mode, "err_cm")][xi]; s = sweeps[(name, mode, "succ")][xi] * 100
        return f"{e.mean():.1f} cm / {s.mean():.0f}%"
    lines = [f"# Zero-shot robustness summary ({a.backend} backend)", "",
             "Cells: mean final error / success rate (<5 cm), averaged over seeds and 100 fixed episodes.", ""]
    for name, xs in (("mass", MASS), ("friction", FRIC), ("delay", DELAY)):
        lines += [f"## {name}", "", "| value | " + " | ".join(LABEL[m] for m in pols) + " |",
                  "|---|" + "---|" * len(pols)]
        for i, x in enumerate(xs):
            lines.append(f"| {x} | " + " | ".join(cell(name, m, i) for m in pols) + " |")
        lines.append("")
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
