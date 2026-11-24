"""Entry point: train, run both MPCs, export results; interactive replay.

    python demo.py run --seed 7 --out results
    python demo.py replay --run results --port 8080
"""
import argparse
import json
import pathlib
import time

import matplotlib
import numpy as np

import estimator
import kinematics
import model
import mpc
import reference
import sysid

N_EVAL_SEEDS = 5
N_TICKS = int(reference.T_END / model.DT_CTRL)  # 240


def closed_loop(models, which, noise):
    """Run one 24 s maneuver with the given predictor; returns log dict."""
    m, d = model.load()
    lo, hi = model.joint_limits(m)
    times, Qref = reference.joint_ref()
    ctrl = mpc.MPC(models[f"F_{which}"], models[f"G_{which}"], lo, hi)
    model.reset(m, d, Qref[0])
    # estimation always uses the coupled model; only the MPC predictor varies
    kf = estimator.KalmanFilter(models["F_coupled"], models["G_coupled"],
                                models["val_resid"], q0=d.qpos + noise[0])
    ctrl.solve(kf.q.copy(), kf.v.copy(), np.zeros(4), Qref[1:1 + mpc.N])  # timing warm-up

    log = {k: [] for k in ("q_true", "q_est", "tooth_true", "tooth_est",
                           "pitch_true", "pitch_est", "u", "call_ms")}
    u = np.zeros(4)
    for k in range(N_TICKS):
        t0 = time.perf_counter()
        if k > 0:
            kf.update(d.qpos + noise[k])          # read encoder, correct
        u = ctrl.solve(kf.q.copy(), kf.v.copy(), u, Qref[k + 1:k + 1 + mpc.N])
        log["call_ms"].append((time.perf_counter() - t0) * 1000)

        p_true, pitch_true = kinematics.fk(d.qpos)
        p_est, pitch_est = kinematics.fk(kf.q)
        for key, val in (("q_true", d.qpos.copy()), ("q_est", kf.q.copy()),
                         ("tooth_true", p_true), ("tooth_est", p_est),
                         ("pitch_true", pitch_true), ("pitch_est", pitch_est),
                         ("u", u)):
            log[key].append(val)

        model.step_ctrl(m, d, u)                  # advance plant 50 steps
        kf.predict(u)                             # predict filter to t[k+1]
    log = {k: np.array(v) for k, v in log.items()}
    log["t"] = times[:N_TICKS]
    log["fallbacks"] = ctrl.fallbacks
    return log


def metrics(logs):
    """Aggregate metrics over seeds for one controller."""
    t = logs[0]["t"]
    on_pass = np.zeros_like(t, dtype=bool)
    for a, b in reference.PASS_WINDOWS:
        on_pass |= (t >= a) & (t <= b)

    est_err = np.concatenate([np.linalg.norm(L["tooth_est"] - L["tooth_true"], axis=1)
                              for L in logs])
    h_err = np.concatenate([L["tooth_true"][on_pass, 2] - reference.GRADE_H for L in logs])
    pitch_err = np.concatenate([L["pitch_true"][on_pass] - reference.PITCH for L in logs])
    call_ms = np.concatenate([L["call_ms"] for L in logs])
    return {
        "est_tooth_rmse_cm": 100 * float(np.sqrt(np.mean(est_err**2))),
        "grade_height_rmse_cm": 100 * float(np.sqrt(np.mean(h_err**2))),
        "worst_dip_below_grade_cm": 100 * float(max(0.0, -h_err.min())),
        "pitch_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(pitch_err**2)))),
        "call_ms_p95": float(np.percentile(call_ms, 95)),
        "calls_over_100ms": int(np.sum(call_ms > 100)),
        "fallbacks": int(sum(L["fallbacks"] for L in logs)),
    }


def make_figure(log_d, log_c, path):
    import matplotlib.pyplot as plt
    t = log_d["t"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for log, name, color in ((log_d, "diagonal MPC", "tab:red"), (log_c, "coupled MPC", "tab:blue")):
        ax.plot(t, 100 * (log["tooth_true"][:, 2] - reference.GRADE_H), color=color, label=name)
    for a, b in reference.PASS_WINDOWS:
        ax.axvspan(a, b, color="0.9", zorder=0)
    for tr in reference.REVERSALS:
        ax.axvline(tr, color="k", ls=":", lw=1)
    ax.set(xlabel="time [s]", ylabel="signed height error [cm]",
           title="Tooth height error (shaded: scored grading passes, dotted: reversals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_video(log_d, log_c, path):
    """Left: side-view arm + tooth paths. Right: height error for both MPCs."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    t = log_c["t"]
    ref = np.array([reference.tooth_ref(tk) for tk in t])
    ref_rz = np.column_stack([np.hypot(ref[:, 0], ref[:, 1]), ref[:, 2]])
    true_rz = np.column_stack([np.hypot(*log_c["tooth_true"][:, :2].T), log_c["tooth_true"][:, 2]])
    est_rz = np.column_stack([np.hypot(*log_c["tooth_est"][:, :2].T), log_c["tooth_est"][:, 2]])

    def arm_points(q):  # joint positions in the reach-height plane
        pts, r, z, a = [(0, kinematics.PIVOT_H)], 0.0, kinematics.PIVOT_H, 0.0
        for L, qj in zip((kinematics.L1, kinematics.L2, kinematics.L3), q[1:]):
            a += qj
            r, z = r + L * np.cos(a), z + L * np.sin(a)
            pts.append((r, z))
        return np.array(pts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ref_rz[:, 0], ref_rz[:, 1], "g--", lw=1, label="desired")
    true_line, = ax1.plot([], [], "b-", lw=1, label="true")
    est_line, = ax1.plot([], [], "r:", lw=1.5, label="estimated")
    arm_line, = ax1.plot([], [], "o-", color="orange", lw=4, ms=5)
    ax1.plot([0], [0], "s", color="gray", ms=12)
    ax1.set(xlim=(-0.5, 5.5), ylim=(-0.2, 4.5), xlabel="reach [m]", ylabel="height [m]",
            title="coupled MPC (side view)", aspect="equal")
    ax1.legend(loc="upper right")

    for log, name, color in ((log_d, "diagonal", "tab:red"), (log_c, "coupled", "tab:blue")):
        ax2.plot(t, 100 * (log["tooth_true"][:, 2] - reference.GRADE_H), color=color, label=name)
    for tr in reference.REVERSALS:
        ax2.axvline(tr, color="k", ls=":", lw=1)
    cursor = ax2.axvline(0, color="0.5")
    ax2.set(xlabel="time [s]", ylabel="signed height error [cm]", title="height error vs grade")
    ax2.legend(loc="upper right")
    fig.tight_layout()

    def frame(k):
        arm = arm_points(log_c["q_true"][k])
        arm_line.set_data(arm[:, 0], arm[:, 1])
        true_line.set_data(true_rz[:k + 1, 0], true_rz[:k + 1, 1])
        est_line.set_data(est_rz[:k + 1, 0], est_rz[:k + 1, 1])
        cursor.set_xdata([t[k]])
        return arm_line, true_line, est_line, cursor

    anim = animation.FuncAnimation(fig, frame, frames=N_TICKS, blit=True)
    anim.save(path, writer=animation.FFMpegWriter(fps=10, bitrate=2000))
    plt.close(fig)


def run(seed, out):
    outdir = pathlib.Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    eval_seeds = [int(s) for s in rng.integers(0, 2**31, N_EVAL_SEEDS)]

    print("training models (calibration seed =", seed, ")")
    models = sysid.train(seed=seed, verbose=True)

    logs = {"diag": [], "coupled": []}
    for es in eval_seeds:
        noise = np.random.default_rng(es).normal(0, estimator.ENC_STD, (N_TICKS, 4))
        for which in ("diag", "coupled"):  # paired: identical noise per seed
            logs[which].append(closed_loop(models, which, noise))
        print(f"  eval seed {es}: done")

    results = {which: metrics(logs[which]) for which in ("diag", "coupled")}
    results["prediction_rmse_rad_s"] = {
        which: dict(zip(model.JOINT_NAMES, np.round(models[f"rmse_{which}"], 4).tolist()))
        for which in ("diag", "coupled")}

    np.savez(outdir / "models.npz",
             **{k: models[k] for k in models if k.startswith(("F_", "G_"))})
    for which in ("diag", "coupled"):
        np.savez(outdir / f"traj_{which}.npz", **logs[which][0])
    json.dump({"seed": seed, "eval_seeds": eval_seeds,
               "encoder_noise_deg": float(np.rad2deg(estimator.ENC_STD)),
               "mpc": {"N": mpc.N, "w_track": mpc.W_TRACK, "w_term": mpc.W_TERM,
                       "w_du": mpc.W_DU, "scale_q": mpc.SCALE_Q.tolist()}},
              open(outdir / "config.json", "w"), indent=2)
    json.dump(results, open(outdir / "metrics.json", "w"), indent=2)

    print("\n=== metrics (5 paired seeds) ===")
    keys = list(results["diag"])
    print(f"{'metric':32s} {'diagonal':>10s} {'coupled':>10s}")
    for k in keys:
        print(f"{k:32s} {results['diag'][k]:10.3f} {results['coupled'][k]:10.3f}")
    print("\nheld-out 1 s velocity prediction RMSE [rad/s]:")
    for which in ("diag", "coupled"):
        print(f"  {which:8s}", results["prediction_rmse_rad_s"][which])

    make_figure(logs["diag"][0], logs["coupled"][0], outdir / "figure.png")
    print("\nencoding video...")
    make_video(logs["diag"][0], logs["coupled"][0], outdir / "video.mp4")
    print("wrote", ", ".join(p.name for p in sorted(outdir.iterdir())))


def replay(run_dir, port):
    """Interactive 3D replay of the saved coupled-MPC rollout (WebViewer)."""
    import mujoco
    from mujoco.studio import launch_web
    from mujoco.studio.viewer_protocol import ViewerConfig

    Q = np.load(pathlib.Path(run_dir) / "traj_coupled.npz")["q_true"]
    m = mujoco.MjModel.from_xml_path(str(model.XML_PATH))
    d = mujoco.MjData(m)
    handle = launch_web.launch(ViewerConfig(gfx="web", http_port=port, open_browser=False))
    handle.set_model(m, data=d)
    print(f"open http://localhost:{port}  (Ctrl-C to stop)")
    k = 0
    while handle.is_running():
        d.qpos[:] = Q[k % len(Q)]
        mujoco.mj_forward(m, d)
        handle.sync()
        time.sleep(model.DT_CTRL)
        k += 1


if __name__ == "__main__":
    matplotlib.use("Agg")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--seed", type=int, default=7)
    pr.add_argument("--out", default="results")
    pp = sub.add_parser("replay")
    pp.add_argument("--run", default="results")
    pp.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    if args.cmd == "run":
        run(args.seed, args.out)
    else:
        replay(args.run, args.port)
