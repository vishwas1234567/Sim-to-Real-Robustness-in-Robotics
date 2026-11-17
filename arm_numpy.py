"""Vectorized NumPy simulator of the 2-link planar arm (fast fallback backend).

Rigid-body dynamics (point mass at the end of each link, angles measured from
the +x axis, gravity along -z):

    M(q) qdd + c(q, qd) + g(q) = tau - damping*qd - coulomb*tanh(qd/eps)

Matches the MuJoCo model in arm_mujoco.py (same masses, lengths, gears, 4
physics substeps of 5 ms per 20 ms control step) except that MuJoCo handles
Coulomb friction ("frictionloss") with its constraint solver instead of a
smooth tanh approximation.
"""
import numpy as np
from arm_common import make_obs, MAX_DELAY

L1 = L2 = 0.5
M1 = M2 = 1.0
G = 9.81
GEAR = np.array([25.0, 12.0])     # max joint torques [N m]
DAMP0 = 0.5                       # viscous damping [N m s/rad]
COULOMB0 = 0.3                    # Coulomb friction [N m]
EPS = 0.05
SUBSTEPS = 4
DT = 0.005


class NumpyArmBatch:
    def reset(self, params, targets, q0):
        self.B = B = len(targets)
        self.m1 = M1 * np.asarray(params["mass"], float)
        self.m2 = M2 * np.asarray(params["mass"], float)
        self.damp = DAMP0 * np.asarray(params["friction"], float)
        self.coul = COULOMB0 * np.asarray(params["friction"], float)
        self.delay = np.asarray(params["delay"]).astype(int)
        self.abuf = np.zeros((MAX_DELAY + 1, B, 2))
        self.idx = np.arange(B)
        self.q = q0.astype(float).copy()
        self.qd = np.zeros((B, 2))
        self.target = targets
        return self._obs()

    def ee_pos(self):
        q1, q2 = self.q[:, 0], self.q[:, 1]
        return np.stack([L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
                         L1 * np.sin(q1) + L2 * np.sin(q1 + q2)], axis=1)

    def _obs(self):
        return make_obs(self.q, self.qd, self.ee_pos(), self.target)

    def _accel(self, q, qd, tau):
        q1, q2 = q[:, 0], q[:, 1]
        w1, w2 = qd[:, 0], qd[:, 1]
        m1, m2 = self.m1, self.m2
        c2, s2 = np.cos(q2), np.sin(q2)
        M11 = (m1 + m2) * L1 ** 2 + m2 * L2 ** 2 + 2 * m2 * L1 * L2 * c2
        M12 = m2 * L2 ** 2 + m2 * L1 * L2 * c2
        M22 = m2 * L2 ** 2
        h = m2 * L1 * L2 * s2
        cor1 = -h * (2 * w1 * w2 + w2 ** 2)
        cor2 = h * w1 ** 2
        g1 = (m1 + m2) * G * L1 * np.cos(q1) + m2 * G * L2 * np.cos(q1 + q2)
        g2 = m2 * G * L2 * np.cos(q1 + q2)
        f1 = self.damp * w1 + self.coul * np.tanh(w1 / EPS)
        f2 = self.damp * w2 + self.coul * np.tanh(w2 / EPS)
        r1 = tau[:, 0] - cor1 - g1 - f1
        r2 = tau[:, 1] - cor2 - g2 - f2
        det = M11 * M22 - M12 ** 2
        return np.stack([(M22 * r1 - M12 * r2) / det,
                         (-M12 * r1 + M11 * r2) / det], axis=1)

    def step(self, action, tau_override=None):
        a = np.clip(action, -1.0, 1.0)
        self.abuf = np.roll(self.abuf, 1, axis=0)       # actuator delay line
        self.abuf[0] = a
        applied = self.abuf[self.delay, self.idx]       # action issued `delay` steps ago
        tau = applied * GEAR if tau_override is None else tau_override
        for _ in range(SUBSTEPS):                       # semi-implicit Euler
            qdd = self._accel(self.q, self.qd, tau)
            self.qd = np.clip(self.qd + DT * qdd, -30.0, 30.0)
            self.q = self.q + DT * self.qd
        dist = np.linalg.norm(self.ee_pos() - self.target, axis=1)
        return self._obs(), dist
