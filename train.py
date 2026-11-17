"""Train a reaching policy with Evolution Strategies (NumPy only, no torch needed).

  python train.py --backend mujoco --mode dr      --seed 0   # Domain Randomization
  python train.py --backend mujoco --mode nominal --seed 0   # baseline, no randomization

Each generation: antithetic (mirrored) perturbations of the policy parameters are
all scored on the SAME K random episodes (common random numbers), fitness is
rank-normalized, and Adam ascends the ES gradient estimate.
In 'dr' mode every episode gets its own random mass / friction / actuator delay.
"""
import argparse, os, time
import numpy as np
from arm_common import (MLPPolicy, make_env, rollout, sample_params, sample_targets,
                        sample_init, tile_conditions, SUCCESS_TOL)


def validate(env, policy, theta, rng, mode, K=64):
    params = sample_params(rng, K, mode)
    r = rollout(env, policy, theta[None], params, sample_targets(rng, K), sample_init(rng, K))
    return r["return"].mean(), r["final_err"].mean(), (r["final_err"] < SUCCESS_TOL).mean()


def train(backend, mode, seed, gens, pop, K, sigma, lr, wd, out_dir, hidden=(32, 32), log_every=10):
    rng = np.random.default_rng(seed)
    env = make_env(backend)
    policy = MLPPolicy(hidden)
    theta = policy.init(rng)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    curve = []
    t0 = time.time()
    for g in range(1, gens + 1):
        eps = rng.standard_normal((pop, policy.n_params))
        cands = np.concatenate([theta + sigma * eps, theta - sigma * eps])      # (2P, D)
        params = sample_params(rng, K, mode)
        targets, q0 = sample_targets(rng, K), sample_init(rng, K)
        tp, tt, tq = tile_conditions(2 * pop, params, targets, q0)
        fit = rollout(env, policy, cands, tp, tt, tq)["return"].reshape(2 * pop, K).mean(1)
        ranks = np.empty(2 * pop); ranks[np.argsort(fit)] = np.arange(2 * pop)
        ranks = ranks / (2 * pop - 1) - 0.5                                     # centred ranks
        grad = ((ranks[:pop] - ranks[pop:]) @ eps) / (pop * sigma) - wd * theta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta = theta + lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
        if g % log_every == 0 or g == 1:
            vr = np.random.default_rng(10_000 + g)       # fixed validation conditions
            n = validate(env, policy, theta, vr, "nominal")
            d = validate(env, policy, theta, np.random.default_rng(20_000 + g), "dr")
            curve.append([g, fit.mean(), *n, *d])
            print(f"[{backend}/{mode}/s{seed}] gen {g:4d}  train_fit {fit.mean():8.2f} | "
                  f"nominal: err {n[1]*100:5.1f} cm succ {n[2]*100:5.1f}% | "
                  f"DR-dist: err {d[1]*100:5.1f} cm succ {d[2]*100:5.1f}% | {time.time()-t0:5.0f}s",
                  flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{backend}_{mode}_seed{seed}.npz")
    np.savez(path, theta=theta, hidden=np.array(hidden), curve=np.array(curve),
             curve_cols=np.array(["gen", "train_fit", "nom_ret", "nom_err", "nom_succ",
                                  "dr_ret", "dr_err", "dr_succ"]))
    print("saved", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy", choices=["numpy", "mujoco"])
    ap.add_argument("--mode", default="dr", choices=["dr", "nominal"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=64, help="mirrored pairs -> 2*pop candidates")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per candidate per generation")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.005)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    train(a.backend, a.mode, a.seed, a.gens, a.pop, a.episodes, a.sigma, a.lr, a.wd, a.out)
