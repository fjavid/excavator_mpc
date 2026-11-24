"""Scripted 24 s grading maneuver: tooth poses and IK joint targets.

Run `python reference.py` to verify the trajectory against joint and
command limits.
"""
import numpy as np

import kinematics
import model

T_END = 24.0
PITCH = np.deg2rad(-45)
SWING2 = np.deg2rad(20)
GRADE_H = 1.0
LIFT_H = 1.3
REVERSALS = (4.0, 19.0)  # pass reversal times, used for scoring/plots
PASS_WINDOWS = ((0.0, 8.0), (15.0, 23.0))  # grade is scored on these


def _blend(a, b, frac):
    """Half-cosine interpolation from a to b, zero endpoint velocity."""
    return a + (b - a) * 0.5 * (1 - np.cos(np.pi * np.clip(frac, 0, 1)))


def tooth_ref(t):
    """Commanded tooth pose (x, y, z, pitch) at time t."""
    t = min(t, T_END)
    if t < 8:            # pass 1 at swing 0: out to 4.2 m and back
        reach, h, sw = 3.2 + 0.5 * (1 - np.cos(2 * np.pi * t / 8)), GRADE_H, 0.0
    elif t < 10:         # lift
        reach, h, sw = 3.2, _blend(GRADE_H, LIFT_H, (t - 8) / 2), 0.0
    elif t < 13:         # swing to 20 deg
        reach, h, sw = 3.2, LIFT_H, _blend(0.0, SWING2, (t - 10) / 3)
    elif t < 15:         # lower back to grade
        reach, h, sw = 3.2, _blend(LIFT_H, GRADE_H, (t - 13) / 2), SWING2
    elif t < 23:         # pass 2 at swing 20 deg
        tl = t - 15
        reach, h, sw = 3.2 + 0.5 * (1 - np.cos(2 * np.pi * tl / 8)), GRADE_H, SWING2
    else:                # hold final pose
        reach, h, sw = 3.2, GRADE_H, SWING2
    return np.array([reach * np.cos(sw), reach * np.sin(sw), h, PITCH])


def joint_ref(dt=model.DT_CTRL, horizon_pad=10):
    """IK joint targets sampled at dt, padded past T_END for the MPC horizon."""
    times = np.arange(0, T_END + (horizon_pad + 1) * dt, dt)
    return times, np.array([kinematics.ik(*tooth_ref(t)) for t in times])


if __name__ == "__main__":
    times, Q = joint_ref()
    lo, hi = model.joint_limits(model.load()[0])
    V = np.diff(Q, axis=0) / model.DT_CTRL

    print(f"{len(times)} samples, t = 0 .. {times[-1]:.1f} s")
    print("start angles [deg]:", np.round(np.rad2deg(Q[0]), 1))
    print("joint range used [deg]:")
    for j, name in enumerate(model.JOINT_NAMES):
        print(f"  {name:7s} [{np.rad2deg(Q[:, j].min()):7.1f}, {np.rad2deg(Q[:, j].max()):7.1f}]"
              f"  limits [{np.rad2deg(lo[j]):7.1f}, {np.rad2deg(hi[j]):7.1f}]")
    print("peak |joint velocity| vs command limit [rad/s]:")
    for j, name in enumerate(model.JOINT_NAMES):
        print(f"  {name:7s} {np.abs(V[:, j]).max():.3f}  limit {model.CMD_LIMIT[j]:.2f}")

    in_range = np.all((Q >= lo) & (Q <= hi))
    in_speed = np.all(np.abs(V) <= model.CMD_LIMIT)
    moved = np.all(Q.max(0) - Q.min(0) > 1e-3)
    print("angles within limits:", in_range)
    print("velocities within command limits:", in_speed)
    print("every joint moves:", moved)
