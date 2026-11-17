"""MuJoCo backend: same 2-link planar arm, same interface as arm_numpy.py.

Each environment in the batch owns its own MjModel so mass / friction can be
randomized per episode by editing model arrays (body_mass, body_inertia,
dof_damping, dof_frictionloss). Actuator delay is implemented as an action
delay line, exactly as in the NumPy backend.

The arm moves in the world x-z plane; the hinge axis is -y so that a positive
joint angle rotates +x towards +z (angle measured from +x, like arm_numpy.py).
"""
import numpy as np
import mujoco
from arm_common import make_obs, MAX_DELAY

XML = """
<mujoco model="planar_arm">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <inertial pos="0.5 0 0" mass="1.0" diaginertia="1e-5 1e-5 1e-5"/>
      <joint name="j1" type="hinge" axis="0 -1 0" damping="0.5" frictionloss="0.3"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" contype="0" conaffinity="0"/>
      <body name="link2" pos="0.5 0 0">
        <inertial pos="0.5 0 0" mass="1.0" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="j2" type="hinge" axis="0 -1 0" damping="0.5" frictionloss="0.3"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" contype="0" conaffinity="0"/>
        <site name="ee" pos="0.5 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" gear="25" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="m2" joint="j2" gear="12" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
SUBSTEPS = 4


class MujocoArmBatch:
    def __init__(self):
        self.models, self.datas = [], []
        ref = mujoco.MjModel.from_xml_string(XML)
        self.base = {"mass": ref.body_mass.copy(), "inertia": ref.body_inertia.copy(),
                     "damping": ref.dof_damping.copy(), "floss": ref.dof_frictionloss.copy()}
        self.site_id = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_SITE, "ee")

    def _ensure(self, B):
        while len(self.models) < B:
            m = mujoco.MjModel.from_xml_string(XML)
            self.models.append(m)
            self.datas.append(mujoco.MjData(m))

    def reset(self, params, targets, q0):
        self.B = B = len(targets)
        self._ensure(B)
        self.delay = np.asarray(params["delay"]).astype(int)
        self.abuf = np.zeros((MAX_DELAY + 1, B, 2))
        self.idx = np.arange(B)
        self.target = targets
        for b in range(B):
            m, d = self.models[b], self.datas[b]
            s, f = float(params["mass"][b]), float(params["friction"][b])
            m.body_mass[:] = self.base["mass"] * s
            m.body_inertia[:] = self.base["inertia"] * s
            m.dof_damping[:] = self.base["damping"] * f
            m.dof_frictionloss[:] = self.base["floss"] * f
            mujoco.mj_setConst(m, d)
            mujoco.mj_resetData(m, d)
            d.qpos[:] = q0[b]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        return self._obs()

    def _state(self):
        q = np.array([d.qpos for d in self.datas[:self.B]])
        qd = np.array([d.qvel for d in self.datas[:self.B]])
        return q, qd

    def ee_pos(self):
        return np.array([d.site_xpos[self.site_id][[0, 2]] for d in self.datas[:self.B]])

    def _obs(self):
        q, qd = self._state()
        return make_obs(q, qd, self.ee_pos(), self.target)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        self.abuf = np.roll(self.abuf, 1, axis=0)
        self.abuf[0] = a
        applied = self.abuf[self.delay, self.idx]
        for b in range(self.B):
            m, d = self.models[b], self.datas[b]
            d.ctrl[:] = applied[b]
            for _ in range(SUBSTEPS):
                mujoco.mj_step(m, d)
        dist = np.linalg.norm(self.ee_pos() - self.target, axis=1)
        return self._obs(), dist
