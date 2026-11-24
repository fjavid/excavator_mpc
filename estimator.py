"""Linear Kalman filter on x = [q, v] with the fitted coupled model.

Run `python estimator.py` for a quick filtering check against truth.
"""
import numpy as np

import model

ENC_STD = np.deg2rad(0.15)  # encoder noise, rad
Q_FLOOR = 1e-8


class KalmanFilter:
    def __init__(self, F, G, val_resid, q0):
        dt = model.DT_CTRL
        I = np.eye(4)
        # x_next = A x + B u:  v_next = Fv + Gu, q_next = q + dt/2 (v + v_next)
        self.A = np.block([[I, 0.5 * dt * (I + F)], [np.zeros((4, 4)), F]])
        self.B = np.vstack([0.5 * dt * G, G])
        self.H = np.hstack([I, np.zeros((4, 4))])
        self.R = ENC_STD**2 * I
        # process noise from held-out one-step velocity residuals
        sv = np.var(val_resid, axis=0) + Q_FLOOR
        self.Qn = np.diag(np.concatenate([(0.5 * dt)**2 * sv + Q_FLOOR, sv]))
        self.x = np.concatenate([q0, np.zeros(4)])
        self.P = np.diag([ENC_STD**2] * 4 + [0.01] * 4)

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Qn

    def update(self, y):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def q(self):
        return self.x[:4]

    @property
    def v(self):
        return self.x[4:]


if __name__ == "__main__":
    import reference
    import sysid

    models = sysid.train(seed=0)
    m, d = model.load()
    _, Qref = reference.joint_ref()
    rng = np.random.default_rng(1)

    model.reset(m, d, Qref[0])
    noise = rng.normal(0, ENC_STD, (80, 4))
    kf = KalmanFilter(models["F_coupled"], models["G_coupled"], models["val_resid"],
                      q0=d.qpos + noise[0])
    eq, ev = [], []
    for k in range(1, 80):  # track the first 8 s of the reference with a crude P-controller
        u = np.clip(5.0 * (Qref[k] - kf.q), -model.CMD_LIMIT, model.CMD_LIMIT)
        model.step_ctrl(m, d, u)
        kf.predict(u)
        kf.update(d.qpos + noise[k])
        eq.append(kf.q - d.qpos)
        ev.append(kf.v - d.qvel)
    raw = np.rad2deg(ENC_STD)
    print(f"raw encoder noise:        {raw:.3f} deg")
    print(f"filtered angle RMSE:      {np.rad2deg(np.sqrt(np.mean(np.square(eq)))):.3f} deg")
    print(f"filtered velocity RMSE:   {np.sqrt(np.mean(np.square(ev))):.4f} rad/s")
