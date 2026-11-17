# Sim-to-Real Robustness: Domain Randomization for a Robot Arm

Train a reaching policy with and without **Domain Randomization (DR)**, then measure how much it
degrades **zero-shot** when **link mass, joint friction and actuator delay** are perturbed at test time.

## Setup
- **Robot / task:** 2-link planar arm in a vertical plane (gravity on). Swing the end-effector to a random target and hold it. 50 Hz control, 3 s episodes. Success = mean error over the last 0.4 s < 5 cm.
- **Policy:** memoryless tanh MLP (10 -> 32 -> 32 -> 2), trained with Evolution Strategies (antithetic sampling, rank-normalized fitness, Adam). NumPy only, no torch needed.
- **Nominal-trained:** physics fixed (mass x1, friction x1, delay 0).
- **DR-trained:** every training episode samples mass x U(0.7, 1.5), friction x logU(0.5, 2.0), delay in {0..3} steps (0-60 ms).
- **Test:** 3 seeds per policy type, 100 fixed episodes per setting, no fine-tuning. 1-D sweeps for each parameter plus 2-D grids (mass x delay, friction x delay).

## Results (numpy backend; mean over 3 seeds)
| Perturbation | Nominal-trained | DR-trained |
|---|---|---|
| Nominal physics | 100% success, 1.6 cm | 99%, 2.8 cm |
| Mass x2.0 (outside train range) | 69%, 15.7 cm | 86%, 5.9 cm |
| Friction x0.25 | 95%, 4.5 cm | 98%, 2.8 cm |
| Delay 1 step (20 ms) | 62%, 7.7 cm | 99%, 2.5 cm |
| Delay 2 steps (40 ms) | 5%, 29.8 cm | 77%, 4.3 cm |
| Delay >= 3 steps | 0% | 0% (error 8 cm at 3 steps vs 58 cm nominal-trained) |

Takeaways
1. **Actuator delay is by far the most damaging perturbation.** One step of delay drops the nominal policy from 100% to 62%; two steps from 100% to 5%.
2. **DR buys large robustness** to delay (<= 2 steps) and to heavy payloads (mass x2), at the cost of ~1 cm worse precision on nominal physics (the usual DR trade-off: a conservative policy).
3. **Friction barely matters** for either policy here (the motors are strong); the nominal policy only suffers at the extremes.
4. **Limit:** a memoryless policy cannot compensate delay >= 3 steps even though 3 was in the DR range. Adding action history / a recurrent policy is the natural next step.

## Files
- `arm_common.py` shared task, randomization ranges, MLP policy, rollout
- `arm_numpy.py` fast vectorized NumPy simulator (used for all results)
- `arm_mujoco.py` MuJoCo version of the same arm (same interface)
- `verify_backends.py` compares the two simulators
- `train.py`, `evaluate.py` training and zero-shot evaluation
- `runs/` trained policies, `logs/` training logs, `results/` figures, `sweeps.csv`, `summary.md`
- `runs_v1/`, `results_v1/` first DR attempt (400 generations, 8 episodes): under-trained, 64% nominal success. Kept for reference; the final DR policies use 1200 generations and 16 episodes per candidate.

## Run
```
python train.py --backend numpy --mode nominal --seed 0 --gens 400
python train.py --backend numpy --mode dr      --seed 0 --gens 1200 --episodes 16
python evaluate.py --backend numpy --runs runs --out results
```
With MuJoCo installed (`pip install mujoco`): run `python verify_backends.py`, then use `--backend mujoco`
(slower, per-environment models, roughly 50-100x slower than the NumPy batch).

## Honest caveats
- **All reported numbers come from the NumPy simulator.** MuJoCo could not be installed in the build sandbox (no network), so `arm_mujoco.py` and `verify_backends.py` are **untested**. Run `verify_backends.py` first; differences should be at the centimetre level.
- The NumPy integrator is semi-implicit Euler at 5 ms: energy drift is a few tenths of a joule over 3 s, acceptable for control experiments but not a precision simulator.
- Only 3 seeds; error bands in the figures are +-1 std over seeds, so treat small gaps (e.g. friction) as noise.
- Arm reaching is not quadruped locomotion or Isaac Lab; it is the smallest task that shows the effect cleanly.
