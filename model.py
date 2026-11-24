"""Load the excavator MJCF, tune velocity-servo gains, step the plant.

Run `python model.py` for the servo pulse-response check.
"""
import pathlib

import mujoco
import numpy as np

XML_PATH = pathlib.Path(__file__).parent / "excavator.xml"

DT_PHYS = 0.002
STEPS_PER_CTRL = 50
DT_CTRL = DT_PHYS * STEPS_PER_CTRL  # 0.1 s

JOINT_NAMES = ["swing", "boom", "stick", "bucket"]
CMD_LIMIT = np.array([0.45, 0.50, 0.65, 0.85])          # rad/s
RESPONSE_T = np.array([0.35, 0.35, 0.25, 0.20])          # s, desired servo time constant
# Representative mid-pass pose (between the given start and reversal angles).
MIDPASS_Q = np.deg2rad([10.0, 38.4, -104.7, 21.3])


def load():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    tune_servos(model, data)
    return model, data


def tune_servos(model, data):
    """kv_i = M_ii(midpass) / response_time_i for each velocity actuator."""
    data.qpos[:] = MIDPASS_Q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)  # densify the joint-space inertia matrix
    kv = np.diag(M) / RESPONSE_T
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = kv[i]
        model.actuator_biasprm[i, 2] = -kv[i]
    return kv


def reset(model, data, q):
    data.qpos[:] = q
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)


def step_ctrl(model, data, u):
    """Apply one clipped velocity command and advance one control interval."""
    data.ctrl[:] = np.clip(u, -CMD_LIMIT, CMD_LIMIT)
    for _ in range(STEPS_PER_CTRL):
        mujoco.mj_step(model, data)


def joint_limits(model):
    return model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()


def _rise_time(model, data, u, joint):
    """63% rise time of qvel[joint] to a step command u (1 s window)."""
    reset(model, data, MIDPASS_Q)
    data.ctrl[:] = u
    target = u[joint]
    for k in range(int(1.0 / DT_PHYS)):
        mujoco.mj_step(model, data)
        if abs(data.qvel[joint]) >= 0.63 * abs(target):
            return (k + 1) * DT_PHYS
    return np.inf


if __name__ == "__main__":
    model, data = load()
    kv = tune_servos(model, data)
    print("servo gains kv:", np.round(kv, 1))

    print("\nindividual pulse 63% rise times (target in parentheses):")
    for j, name in enumerate(JOINT_NAMES):
        u = np.zeros(4)
        u[j] = 0.3 * CMD_LIMIT[j]
        t = _rise_time(model, data, u, j)
        print(f"  {name:7s} {t:.3f} s  ({RESPONSE_T[j]:.2f})")

    print("\nsimultaneous pulse:")
    u = 0.3 * CMD_LIMIT * np.array([1, 1, -1, 1])
    for j, name in enumerate(JOINT_NAMES):
        t = _rise_time(model, data, u, j)
        print(f"  {name:7s} {t:.3f} s")
