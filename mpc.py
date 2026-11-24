"""Receding-horizon MPC over the fitted velocity model (CVXPY + OSQP).

The problem is built once per model with Parameters; each tick updates
them and re-solves warm-started. Run `python mpc.py` for a smoke test.
"""
import time

import cvxpy as cp
import numpy as np

import model

N = 10              # horizon steps (1 s at 10 Hz)
W_TRACK = 1.0
W_TERM = 5.0
W_DU = 0.02
SCALE_Q = np.array([0.05, 0.05, 0.05, 0.05])  # rad, per-joint angle error scale
MARGIN = 0.02       # rad, keep-out from joint stops


class MPC:
    def __init__(self, F, G, lo, hi):
        dt = model.DT_CTRL
        self.U = cp.Variable((N, 4))
        self.q0 = cp.Parameter(4)
        self.v0 = cp.Parameter(4)
        self.u_prev = cp.Parameter(4)
        self.Qref = cp.Parameter((N, 4))

        cost, cons = 0, [cp.abs(self.U) <= model.CMD_LIMIT]
        q, v, u_last = self.q0, self.v0, self.u_prev
        for k in range(N):
            u = self.U[k]
            v_next = F @ v + G @ u
            q = q + 0.5 * dt * (v + v_next)
            v = v_next
            w = W_TERM if k == N - 1 else W_TRACK
            cost += w * cp.sum_squares((q - self.Qref[k]) / SCALE_Q)
            cost += W_DU * cp.sum_squares((u - u_last) / model.CMD_LIMIT)
            cons += [q >= lo + MARGIN, q <= hi - MARGIN]
            u_last = u
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self.fallbacks = 0
        self.solve_times = []

    def solve(self, q0, v0, u_prev, qref):
        """Returns the first command; zero-speed fallback on solver failure."""
        t0 = time.perf_counter()
        self.q0.value, self.v0.value = q0, v0
        self.u_prev.value, self.Qref.value = u_prev, qref
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            ok = self.prob.status in ("optimal", "optimal_inaccurate")
        except cp.SolverError:
            ok = False
        self.solve_times.append(time.perf_counter() - t0)
        if not ok:
            self.fallbacks += 1
            return np.zeros(4)
        return np.clip(self.U.value[0], -model.CMD_LIMIT, model.CMD_LIMIT)


if __name__ == "__main__":
    import reference
    import sysid

    models = sysid.train(seed=0)
    m, d = model.load()
    lo, hi = model.joint_limits(m)
    _, Qref = reference.joint_ref()
    ctrl = MPC(models["F_coupled"], models["G_coupled"], lo, hi)

    model.reset(m, d, Qref[0])
    u = np.zeros(4)
    errs = []
    for k in range(240):  # perfect-state smoke test over the full maneuver
        u = ctrl.solve(d.qpos.copy(), d.qvel.copy(), u, Qref[k + 1:k + 1 + N])
        model.step_ctrl(m, d, u)
        errs.append(d.qpos - Qref[k + 1])
    st = np.array(ctrl.solve_times) * 1000
    print(f"joint tracking RMSE [deg]: {np.round(np.rad2deg(np.sqrt(np.mean(np.square(errs), 0))), 3)}")
    print(f"solve time p50/p95 [ms]: {np.percentile(st, 50):.1f} / {np.percentile(st, 95):.1f}")
    print(f"fallbacks: {ctrl.fallbacks}/240")
