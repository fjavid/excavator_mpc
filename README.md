# Four-joint excavator grading with MuJoCo and MPC

A parked excavator traces a 1 m horizontal grade with its bucket tooth,
reverses, lifts, swings 20°, and repeats. Joint states are estimated from
noisy encoders with a Kalman filter; a constrained MPC tracks IK joint
targets using a velocity model learned from simulated calibration runs.
The experiment: does learning the *coupling* between joints (full 4x4
`v_next = Fv + Gu`) beat four independent per-joint models?

## Run

```bash
python -m pip install -r requirements.txt
python demo.py run --seed 7 --out results      # train, evaluate, export (~30 s)
python demo.py replay --run results --port 8080  # interactive 3D replay (WebViewer)
# on a laptop, if the run lives on a VM:
gcloud compute ssh VM_NAME --zone=ZONE -- -N -L 8080:localhost:8080
```

`results/` then contains the fitted models (`models.npz`), resolved config
and evaluation seeds (`config.json`), one trajectory log per controller
(`traj_*.npz`), `metrics.json`, `figure.png`, and `video.mp4`.

Each module is also runnable as its own check:
`model.py` (servo pulse response), `kinematics.py` (FK vs MuJoCo site,
IK checkpoint angles), `reference.py` (limit checks), `sysid.py`
(held-out prediction RMSE), `estimator.py` (filter vs truth), `mpc.py`
(perfect-state tracking smoke test).

## Files

| File | Contents |
|---|---|
| `excavator.xml` | MJCF: swing/boom/stick/bucket chain, velocity actuators, `tooth` site |
| `model.py` | plant loading, inertia-based servo gain tuning, control-rate stepping |
| `kinematics.py` | analytic FK and IK (law of cosines, negative stick branch) |
| `reference.py` | scripted 24 s maneuver -> tooth poses -> IK joint targets |
| `sysid.py` | calibration runs, ridge fit of diagonal and coupled velocity models |
| `estimator.py` | 8-state linear KF on `[q, v]` using the coupled model |
| `mpc.py` | CVXPY/OSQP receding-horizon controller, built once, warm-started |
| `demo.py` | end-to-end run, metrics, figure, MP4, WebViewer replay |

## Results (seed 7, five paired evaluation seeds)

| Metric | Diagonal MPC | Coupled MPC |
|---|---|---|
| Held-out 1 s velocity prediction RMSE, bucket [rad/s] | 0.488 | 0.269 |
| (swing / boom / stick are within 0.003-0.03 of each other) | | |
| True-vs-estimated tooth position RMSE [cm] | 1.06 | 1.06 |
| Grade height RMSE, passes only [cm] | 1.15 | 0.87 |
| Worst dip below grade [cm] | 3.98 | 3.16 |
| Pitch RMSE, passes only [deg] | 0.86 | 0.34 |
| 95th-pct controller call [ms] | 3.2 | 2.8 |
| Calls over 100 ms / solver fallbacks | 0 / 0 | 0 / 0 |

Both controllers solved every QP; no fallbacks and no joint-limit
excursions. Estimation quality is identical by construction (both use the
coupled model in the filter), so the differences above are purely the MPC
prediction matrices.

**Observed difference at a reversal.** Around the 4 s reversal the boom
and stick decelerate and reverse. Their motion drags the bucket joint
through reaction torques the diagonal model cannot represent, so the
diagonal MPC only reacts after the error appears: pitch error peaks at
3.3° and height error at 2.9 cm just after the reversal. The coupled MPC
predicts the drag and pre-compensates, holding the same reversal to 0.5°
and 1.3 cm. The same pattern repeats at 19 s. Interestingly, the coupled
model is a slightly *worse* predictor for the stick joint, yet still
controls better: what matters for grading is the bucket, whose prediction
error it halves.

**Limitation of velocity servos.** Real excavator joints are driven by
hydraulic cylinders through linkages: available torque depends on
cylinder geometry and pressure, responds asymmetrically (gravity assists
lowering), saturates, and has valve deadband and flow-coupling between
functions sharing a pump. A gravity-compensated velocity servo with a
constant gain hides all of that; it can command any torque instantly and
symmetrically. The learned-model + MPC structure would carry over, but
the fitted dynamics and achievable accuracy here are optimistic relative
to a real machine.

## Scope

Free-space motion (no soil contact), known stationary base, illustrative
dimensions. The controller sees only filter output, reference, and its
own past commands; simulator ground truth is used for calibration and
scoring only. Calibration (30 runs: 20 train / 5 val / 5 test, split by
seed) is disjoint from the five paired evaluation seeds, which share
pre-generated encoder noise between the two controllers.
