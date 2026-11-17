"""Check that the MuJoCo model and the NumPy simulator agree (run where `mujoco` is installed).

Drives both with the same open-loop torque sequence and reports the end-effector
difference. Small differences are expected (MuJoCo's Coulomb friction is solved as a
constraint, the NumPy one uses a tanh approximation) but they should stay at the
centimetre level, otherwise something is wrong with the model.

  python verify_backends.py
"""
import numpy as np
from arm_common import nominal_params, sample_targets, EP_LEN
from arm_numpy import NumpyArmBatch
from arm_mujoco import MujocoArmBatch

rng = np.random.default_rng(0)
B = 8
q0 = np.array([-0.8, 0.5]) + 0.1 * rng.standard_normal((B, 2))
targets = sample_targets(rng, B)
acts = 0.4 * np.sin(np.linspace(0, 6, EP_LEN))[:, None, None] * rng.uniform(0.5, 1, (1, B, 2))
for label, p in [("nominal", {}), ("mass x1.5", {"mass": 1.5}), ("friction x2", {"friction": 2.0})]:
    params = nominal_params(B)
    for k, v in p.items():
        params[k] = np.full(B, v)
    a, b = NumpyArmBatch(), MujocoArmBatch()
    a.reset(params, targets, q0); b.reset(params, targets, q0)
    worst = 0.0
    for t in range(EP_LEN):
        a.step(acts[t]); b.step(acts[t])
        worst = max(worst, np.linalg.norm(a.ee_pos() - b.ee_pos(), axis=1).max())
    print(f"{label:12s} max end-effector difference over episode: {worst*100:.2f} cm")
