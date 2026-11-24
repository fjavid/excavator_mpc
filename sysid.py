"""Calibration runs and fitting of the diagonal / coupled velocity models.

Model: v_next = F @ v + G @ u at the 10 Hz control rate.
Run `python sysid.py` to collect data, fit both models, and print the
held-out one-second rollout RMSE comparison.
"""
import numpy as np

import model
import reference

N_RUNS = 30
RUN_T = 8.0
SPLIT = (20, 5, 5)  # train / val / test runs
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0)
STOP_MARGIN = 0.12  # rad, steer commands away from joint stops


def collect_runs(seed=0):
    """30 runs of bounded randomized commands; returns list of (V, U) logs."""
    m, d = model.load()
    lo, hi = model.joint_limits(m)
    _, Qref = reference.joint_ref()
    rng = np.random.default_rng(seed)
    n_ticks = int(RUN_T / model.DT_CTRL)
    runs = []
    for _ in range(N_RUNS):
        q0 = Qref[rng.integers(len(Qref))]  # start near a maneuver pose
        model.reset(m, d, q0)
        u = np.zeros(4)
        next_change = 0
        V, U = np.zeros((n_ticks + 1, 4)), np.zeros((n_ticks, 4))
        V[0] = d.qvel
        for k in range(n_ticks):
            if k >= next_change:  # new random command every 0.3-0.7 s
                u = rng.uniform(-model.CMD_LIMIT, model.CMD_LIMIT)
                next_change = k + rng.integers(3, 8)
            # steer away from joint stops instead of resetting
            u = np.where(d.qpos > hi - STOP_MARGIN, np.minimum(u, 0), u)
            u = np.where(d.qpos < lo + STOP_MARGIN, np.maximum(u, 0), u)
            U[k] = u
            model.step_ctrl(m, d, u)
            V[k + 1] = d.qvel
        runs.append((V, U))
    return runs


def _stack(runs):
    """Aligned (v[k], u[k]) -> v[k+1] triples from whole runs."""
    X = np.vstack([np.hstack([V[:-1], U]) for V, U in runs])
    Y = np.vstack([V[1:] for V, U in runs])
    return X, Y


def fit(runs, coupled, alpha):
    """Ridge fit with zero intercept; returns (F, G) as 4x4 matrices."""
    X, Y = _stack(runs)
    if coupled:
        W = np.linalg.solve(X.T @ X + alpha * np.eye(8), X.T @ Y).T  # 4x8
        return W[:, :4], W[:, 4:]
    F, G = np.zeros((4, 4)), np.zeros((4, 4))
    for j in range(4):
        Xj = X[:, [j, 4 + j]]
        w = np.linalg.solve(Xj.T @ Xj + alpha * np.eye(2), Xj.T @ Y[:, j])
        F[j, j], G[j, j] = w
    return F, G


def rollout_rmse(F, G, runs, horizon=10):
    """Per-joint RMSE of 1 s open-loop velocity rollouts [rad/s]."""
    err = []
    for V, U in runs:
        for k0 in range(len(U) - horizon):
            v = V[k0].copy()
            for k in range(k0, k0 + horizon):
                v = F @ v + G @ U[k]
                err.append(v - V[k + 1])
    return np.sqrt(np.mean(np.square(err), axis=0))


def residuals(F, G, runs):
    """One-step velocity prediction residuals (for KF process noise)."""
    X, Y = _stack(runs)
    return Y - X[:, :4] @ F.T - X[:, 4:] @ G.T


def train(seed=0, verbose=False):
    """Collect data, pick alpha on validation runs, fit both models.

    Returns dict with F/G for both models, val residuals of the coupled
    model, and per-joint test rollout RMSE for both.
    """
    runs = collect_runs(seed)
    n_tr, n_val, _ = SPLIT
    tr, val, test = runs[:n_tr], runs[n_tr:n_tr + n_val], runs[n_tr + n_val:]

    out = {}
    for name, coupled in (("diag", False), ("coupled", True)):
        best = min(ALPHAS, key=lambda a: rollout_rmse(*fit(tr, coupled, a), val).mean())
        F, G = fit(tr, coupled, best)
        assert np.max(np.abs(np.linalg.eigvals(F))) < 1, f"{name} model unstable"
        out[f"F_{name}"], out[f"G_{name}"] = F, G
        out[f"rmse_{name}"] = rollout_rmse(F, G, test)
        if verbose:
            print(f"{name:8s} alpha={best:g}  test 1 s rollout RMSE [rad/s]: "
                  + "  ".join(f"{n}={r:.4f}" for n, r in zip(model.JOINT_NAMES, out[f"rmse_{name}"])))
    out["val_resid"] = residuals(out["F_coupled"], out["G_coupled"], val)
    return out


if __name__ == "__main__":
    train(seed=0, verbose=True)
