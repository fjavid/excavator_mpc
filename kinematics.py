"""Analytic FK/IK for the four-joint excavator.

q = [swing, boom, stick, bucket] in radians. Tooth pose is world (x, y, z)
plus pitch (bucket link angle from horizontal, in the reach-height plane).

Run `python kinematics.py` to check FK against the MuJoCo tooth site and
IK against the plan's checkpoint angles.
"""
import numpy as np

PIVOT_H = 1.5          # boom pivot height above world ground [m]
L1, L2, L3 = 3.0, 2.0, 0.8  # boom, stick, bucket link lengths [m]


def fk(q):
    """World tooth position and pitch for joint angles q."""
    swing, boom, stick, bucket = q
    a1 = boom
    a2 = boom + stick
    a3 = boom + stick + bucket
    reach = L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)
    z = PIVOT_H + L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3)
    x = reach * np.cos(swing)
    y = reach * np.sin(swing)
    return np.array([x, y, z]), a3


def ik(x, y, z, pitch):
    """Joint angles for a world tooth pose. Raises ValueError if unreachable."""
    swing = np.arctan2(y, x)
    reach = np.hypot(x, y)
    # wrist = bucket hinge position in the reach-height plane
    rw = reach - L3 * np.cos(pitch)
    zw = (z - PIVOT_H) - L3 * np.sin(pitch)
    d = (rw**2 + zw**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(d) > 1.0:
        raise ValueError(f"unreachable tooth pose ({x:.2f},{y:.2f},{z:.2f},{pitch:.2f})")
    stick = -np.arccos(d)  # negative stick-angle branch
    boom = np.arctan2(zw, rw) - np.arctan2(L2 * np.sin(stick), L1 + L2 * np.cos(stick))
    bucket = pitch - boom - stick
    return np.array([swing, boom, stick, bucket])


if __name__ == "__main__":
    import mujoco
    import model

    # 1. IK at the plan's checkpoint poses
    q0 = ik(3.2, 0.0, 1.0, np.deg2rad(-45))
    q4 = ik(4.2, 0.0, 1.0, np.deg2rad(-45))
    print("t=0 angles [deg]:", np.round(np.rad2deg(q0), 1), " expected [0, 42.4, -120.3, 32.9]")
    print("t=4 angles [deg]:", np.round(np.rad2deg(q4), 1), " expected [0, 34.4, -89.0, 9.6]")

    # 2. FK vs MuJoCo tooth site across representative poses
    m, d = model.load()
    site = m.site("tooth").id
    rng = np.random.default_rng(0)
    lo, hi = model.joint_limits(m)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(lo, hi)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        pos, _ = fk(q)
        worst = max(worst, np.linalg.norm(pos - d.site_xpos[site]))
    print(f"max |fk - mujoco site| over 200 random poses: {worst * 1000:.4f} mm (must be < 1 mm)")

    # 3. IK -> FK round trip
    p, pitch = fk(q0)
    q_rt = ik(*p, pitch)
    print("round-trip error:", np.max(np.abs(q_rt - q0)))

    # 4. unreachable pose is reported
    try:
        ik(7.0, 0.0, 1.0, 0.0)
    except ValueError as e:
        print("unreachable correctly raised:", e)
