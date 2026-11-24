# Four-joint excavator grading with MuJoCo and model predictive control

## Assignment

Build a runnable Python project aiming to teach the user the application of MCP and to prepare them for interviews in robotics companies. A parked excavator should trace a one-meter horizontal grade with its bucket tooth, reverse, lift, swing 20°, and repeat. Estimate the tooth pose from noisy joint encoders. Learn actuator dynamics from simulated calibration runs and use the learned model in a constrained model predictive controller (MPC).

The experiment asks whether learning coupling between the joints improves prediction and grading accuracy. Deliver a runnable demo, an interview video, and a small table of measured results. Use native MuJoCo physics on the CPU of the M4 Macbook and render the 3D replay. Scope the demo to free-space motion, gravity-compensated velocity servos, and a known stationary base. Its dimensions and actuator parameters are illustrative. The learning task is supervised system identification.

### 1. Build a four-joint MuJoCo model

We will use the virtual environment set up on the root of this repo (.venv/), its native Python3.12, `mujoco==4.0.0`, NumPy, scikit-learn, CVXPY, OSQP, and Matplotlib. Use FFmpeg for MP4 export. Model the machine in a self-contained `excavator.xml` using primitive geometry. Mark the tip with a named MuJoCo site, `tooth`. Use SI units and set the MJCF compiler angle unit to radians.

| Joint order | Positive rotation | Starting angle limits | Command limit |
|---|---|---|---|
| 0. House swing | About world +Z, counterclockwise from above | −90° to 90° | ±0.45 rad/s |
| 1. Boom | Raises a horizontal link | 10° to 75° | ±0.50 rad/s |
| 2. Stick | Relative to boom | −150° to −45° | ±0.65 rad/s |
| 3. Bucket curl | Relative to stick | −65° to 75° | ±0.85 rad/s |

Align the links along local +X and the three pitch hinges along local −Y. Model the boom pivot directly on the swing axis, 1.5 m above world ground. Use lengths of 3.0 m from boom pivot to stick pivot, 2.0 m from stick pivot to bucket hinge, and 0.8 m from bucket hinge to tooth. Draw the target grade as a visual line. Keep the moving links collision-free for this experiment.

Use four unit-geared MuJoCo velocity actuators and advance the plant with `mj_step`. Give each moving body positive mass and inertia from simple geoms and enable gravity compensation. Use an implicit integrator such as `implicitfast`, a 0.002 s physics step, and 50 physics steps per 0.1 s control interval. As an initial servo tuning, evaluate the joint-space inertia matrix at mid-pass and set each velocity gain to its diagonal inertia divided by the desired response time. Start with 0.35, 0.35, 0.25, and 0.20 s for swing, boom, stick, and bucket. Verify the resulting response with individual and simultaneous command pulses. MuJoCo supplies the actual coupled motion. Keep the plant distinct from the fitted predictor used by MPC.

### 2. Implement and verify the reference

Write `fk(q)` for world tooth position and pitch and `ik(x, y, z, pitch)` for four joint angles. For IK, compute swing from `atan2(y,x)` and reach from `hypot(x,y)`. Subtract pivot height from world height, then subtract the last link vector `0.8 * [cos(pitch), sin(pitch)]` in the reach–height plane. Solve the remaining two-link arm by the law of cosines, using the negative stick-angle branch. Bucket curl is pitch minus boom minus stick. Report unreachable targets explicitly.

| Time | Commanded tooth motion |
|---|---|
| 0–8 s | At swing 0° and pitch −45°, set reach to `3.2 + 0.5*(1 − cos(2πt/8))` m and height to 1.0 m in world coordinates. Reverse at 4 s. |
| 8–10 s | At reach 3.2 m, lift from 1.0 to 1.3 m. |
| 10–13 s | Hold reach, height, and pitch while swinging from 0° to 20°. |
| 13–15 s | Lower back to grade at the new swing angle. |
| 15–23 s | Repeat the outward and return pass using local time `t − 15`. |
| 23–24 s | Hold the final pose. Extend this hold when filling the MPC horizon. |

Use a half-cosine interpolation for each transition so endpoint velocity is zero. Initialize at the first reference pose with zero speed. At the first reversal the three pitch-joint angles should be approximately **34.4°, −89.0°, 9.6°**. At the start they should be **42.4°, −120.3°, 32.9°**. Verify the full trajectory against joint and command limits. Independently check the analytic FK point against MuJoCo’s world-space tooth site to within 1 mm across representative poses. This catches sign, link-offset, and pivot-height errors that an FK–IK round trip can miss.

### 3. Collect calibration data and train two models

Start with 30 independent eight-second runs, initialized near different poses on the maneuver. Use bounded, independently randomized joint-speed commands, changing every 0.3–0.7 s. Include simultaneous commands and sign changes. Keep excitation inside the intended operating range and reset before reaching a joint stop. Split whole trajectories into 20 training, 5 validation, and 5 test runs by seed.

Log aligned triples `(v[k], u[k], v[k+1])` at the **control** rate. Simulator velocities are valid offline training labels. Fit `v_next = F @ v + G @ u` by ridge regression, with zero intercept. For the diagonal model, fit one output from that joint’s own velocity and command. For the coupled model, fit each output from all four velocities and commands. Validate regularization and check that zero-input predicted velocities decay. On untouched test trajectories, compare one-second open-loop rollout RMSE in rad/s using identical initial states and ten recorded commands. Record per-joint values so a weak bucket prediction is visible.

**checkpoint:** the first pass animates correctly, pulse commands produce measurable lag, and the held-out prediction comparison runs. A clipped proportional controller with reference-velocity feedforward is a useful first closed-loop smoke test.

### 4. Estimate the joint state from encoders

Use an eight-state linear Kalman filter with `x = [q, v]`. Generate encoder angles by adding independently seeded Gaussian noise to simulated joint angles, starting at 0.15° standard deviation. The filter receives the four measured angles and the preceding command. Use the coupled model for prediction in **both** controller experiments. For each prediction step use

```text
v_next = F @ v + G @ u
q_next = q + 0.5 * dt * (v + v_next)
encoder_prediction = q_next
```

This is a linear state update. Use the matching matrices in the covariance prediction, an angle-only observation matrix, and measurement covariance based on the configured encoder noise. Estimate process-noise scales from held-out validation residuals and add a small positive diagonal floor. Apply the same world-frame FK used by the reference to the corrected angles to obtain the estimated tooth position and pitch. The parked base has a known pose. The online controller receives only the filter output, reference trajectory, and previous commands. Reserve simulator velocities and tooth poses for calibration or scoring.

### 5. Implement MPC and compare predictors

At 10 Hz, optimize ten four-joint commands covering one second. Start every solve at the corrected filter state. Use the fitted joint-speed and trapezoidal angle update above. Track the **next** ten IK-derived angle targets, penalize changes from the previous command, and impose the listed joint-angle and speed-command bounds. Normalize angle errors and command changes by per-joint scales and use a terminal tracking weight. Build the CVXPY problem once per model, update parameters each tick, warm-start OSQP, apply only the first command, and hold it through the next 50 physics steps.

```text
read encoder at t[k] → correct estimate → solve MPC → apply u[k]
advance MuJoCo 50 steps → predict filter to t[k+1] with u[k] → repeat
```

Log solver status, elapsed time around the entire controller call, any fallback, and the actual command. If a solve fails, use a bounded zero-speed command for that interval and flag the run. For timing, warm up first and turn off rendering. Report solver failures and physical limit excursions separately from predicted feasibility.

Run the **diagonal-predictor MPC** and **coupled-predictor MPC** against the same MuJoCo plant. Give them the same filter parameters, controller weights, reference, initial pose, and pre-generated encoder noise for each of five paired seeds. The MPC prediction matrices are the experimental variable. Use validation runs to set weights before evaluating these seeds. Report the measured difference even if learning a coupled predictor helps prediction more than it helps control.

### 6. Deliver the interview demo

| Output | Contents |
|---|---|
| Metrics table | Held-out one-second velocity prediction RMSE, true-versus-estimated tooth position RMSE in cm, true tooth-height RMSE and worst dip below grade in cm, pitch RMSE in degrees, 95th-percentile controller-call time, and count above 100 ms. Score grade on the two horizontal passes only. Report fallback counts over the whole maneuver. |
| Short MP4 | Use a saved rollout. Show the excavator and desired/true/estimated tooth path on one side. On the other, overlay signed height error for both controllers, marking the reversals at 4 and 19 s. Use Matplotlib and FFmpeg so export works on the headless VM. |
| Interactive 3D replay | Load the same MJCF and saved joint angles in MuJoCo Studio’s Python WebViewer, set the four joint positions, run `mj_forward` and sync the viewer for each logged frame. Connect from the laptop through SSH port forwarding. Use the standalone viewer to check the geometry and the recorded rollout to demonstrate the controller. |

Provide a README, pinned dependencies, the MJCF model, and a simple Python entry point. Document commands equivalent to the following. The `run` command should also save the evaluation seed list derived from `--seed`. The output directory should contain the fitted models, resolved configuration and seeds, two trajectory logs, metrics, figure, and video.

```bash
python -m pip install -r requirements.txt
python -m excavator.demo run --seed 7 --out results      # train, evaluate and export
python -m excavator.demo replay --run results --port 8080
gcloud compute ssh VM_NAME --zone=ZONE -- -N -L 8080:localhost:8080  # on laptop
```

### Completion checks

- A fresh headless CPU run finishes, produces finite trajectories and metrics for both controllers, and generates a playable MP4.
- FK agrees with the simulator site, IK reconstructs the requested poses, every joint moves somewhere in the maneuver, and commanded speeds stay within bounds.
- Holding seed and configuration fixed reproduces the trajectories within numerical tolerance. Calibration and final scoring use disjoint runs. The estimator and MPC have no access to scoring ground truth.
- Report nominal solve success, missed deadlines, and any excursions. Treat lower grade error or a faster solve as a measured outcome. In the README, explain one observed difference at a reversal and one limitation of using velocity servos to approximate excavator actuation.

### Implementation references

- [MuJoCo 4.0 Python bindings](https://mujoco.readthedocs.io/en/4.0.0/python.html) and [MJCF velocity actuators](https://mujoco.readthedocs.io/en/4.0.0/XMLreference.html#actuator-velocity)
- [MuJoCo Studio remote WebViewer and custom loop](https://mujoco.readthedocs.io/en/4.0.0/studio/web_viewers.html)
- [OSQP receding-horizon MPC example](https://osqp.org/docs/examples/mpc.html)
- [Full-scale grading research for interview context](https://roboticsproceedings.org/rss22/p111.html)