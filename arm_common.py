"""Shared definitions for the sim-to-real robustness project.

Task: a 2-link planar arm in a vertical plane (gravity acts) must swing its
end-effector to a random target and hold it there.

Three physical quantities are randomized during training (Domain Randomization)
and perturbed at test time (zero-shot robustness test):

  mass      multiplier on link masses (and inertias)
  friction  multiplier on joint viscous damping and Coulomb friction
  delay     actuator delay, in control steps (1 step = 20 ms)

Both simulator backends (arm_numpy.py, arm_mujoco.py) expose the same batched
interface, so training / evaluation code is backend-agnostic:

  env.reset(params, targets, q0) -> obs (B, 10)
  env.step(action (B, 2))        -> obs (B, 10), dist (B,)
  env.ee_pos()                   -> (B, 2)   end-effector (x, z)
"""
import numpy as np

OBS_DIM, ACT_DIM = 10, 2
EP_LEN = 150            # control steps -> 3 s
DT_CTRL = 0.02          # 50 Hz control
MAX_DELAY = 10
ACT_COST = 0.002
FINAL_WINDOW = 20       # error is averaged over the last 20 steps (0.4 s)
SUCCESS_TOL = 0.05      # metres

# Training-time randomization ranges (Domain Randomization).
TRAIN_RANGES = {"mass": (0.7, 1.5), "friction": (0.5, 2.0), "delay": (0, 3)}
NOMINAL = {"mass": 1.0, "friction": 1.0, "delay": 0}


def nominal_params(B):
    return {"mass": np.full(B, 1.0), "friction": np.full(B, 1.0),
            "delay": np.zeros(B, dtype=int)}


def sample_params(rng, B, mode):
    """mode='nominal' -> no randomization; mode='dr' -> sample TRAIN_RANGES."""
    if mode == "nominal":
        return nominal_params(B)
    (m0, m1), (f0, f1), (d0, d1) = (TRAIN_RANGES["mass"], TRAIN_RANGES["friction"],
                                    TRAIN_RANGES["delay"])
    return {"mass": rng.uniform(m0, m1, B),
            "friction": np.exp(rng.uniform(np.log(f0), np.log(f1), B)),  # log-uniform
            "delay": rng.integers(d0, d1 + 1, B)}


def sample_targets(rng, B):
    r = rng.uniform(0.4, 0.9, B)
    phi = rng.uniform(-0.5, 1.2, B)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def sample_init(rng, B):
    return np.array([-0.8, 0.5]) + 0.1 * rng.standard_normal((B, 2))


def make_obs(q, qd, ee, target):
    return np.concatenate([np.cos(q[:, :1]), np.sin(q[:, :1]),
                           np.cos(q[:, 1:]), np.sin(q[:, 1:]),
                           0.2 * qd, target, target - ee], axis=1)


def tile_conditions(P, params, targets, q0):
    """Repeat one set of K episode conditions for each of P policies (index p*K+k)."""
    tp = {k: np.tile(v, P) for k, v in params.items()}
    return tp, np.tile(targets, (P, 1)), np.tile(q0, (P, 1))


class MLPPolicy:
    """Batched tanh MLP; evaluates P parameter vectors on K episodes each."""

    def __init__(self, hidden=(32, 32)):
        self.sizes = [OBS_DIM, *hidden, ACT_DIM]
        self.n_params = sum(i * o + o for i, o in zip(self.sizes[:-1], self.sizes[1:]))

    def init(self, rng):
        parts = []
        for n, (i, o) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            scale = (1.0 / np.sqrt(i)) * (0.1 if n == len(self.sizes) - 2 else 1.0)
            parts += [rng.standard_normal(i * o) * scale, np.zeros(o)]
        return np.concatenate(parts)

    def act(self, thetas, obs):
        """thetas (P, D), obs (P, K, OBS_DIM) -> actions (P, K, ACT_DIM) in [-1, 1]."""
        P = thetas.shape[0]
        x, pos = obs, 0
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            W = thetas[:, pos:pos + i * o].reshape(P, i, o); pos += i * o
            b = thetas[:, pos:pos + o]; pos += o
            x = np.tanh(np.einsum("pki,pio->pko", x, W) + b[:, None, :])
        return x


def rollout(env, policy, thetas, params, targets, q0, record=False):
    """Run P policies x K episodes (B = P*K envs, index p*K + k). Returns dict."""
    P = thetas.shape[0]
    B = len(targets)
    K = B // P
    obs = env.reset(params, targets, q0)
    ret = np.zeros(B)
    dist_hist = np.zeros((EP_LEN, B))
    ee_hist = []
    for t in range(EP_LEN):
        a = policy.act(thetas, obs.reshape(P, K, -1)).reshape(B, -1)
        obs, dist = env.step(a)
        ret += -dist - ACT_COST * (a ** 2).sum(1)
        dist_hist[t] = dist
        if record:
            ee_hist.append(env.ee_pos().copy())
    final = dist_hist[-FINAL_WINDOW:].mean(0)
    return {"return": ret, "final_err": final, "success": final < SUCCESS_TOL,
            "dist_hist": dist_hist, "ee_hist": np.array(ee_hist) if record else None}


def make_env(backend):
    if backend == "numpy":
        from arm_numpy import NumpyArmBatch
        return NumpyArmBatch()
    if backend == "mujoco":
        from arm_mujoco import MujocoArmBatch
        return MujocoArmBatch()
    raise ValueError(backend)
