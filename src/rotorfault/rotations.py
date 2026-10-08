"""Small quaternion / rotation helpers (scalar-first unit quaternions, body-to-world)."""

from __future__ import annotations

import math

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns a quaternion with non-negative scalar part."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_exp(v) -> np.ndarray:
    """Quaternion of the rotation vector ``v`` (rad)."""
    th = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if th < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
    s = math.sin(0.5 * th) / th
    return np.array([math.cos(0.5 * th), s * v[0], s * v[1], s * v[2]])


def quat_log(q) -> np.ndarray:
    """Rotation vector of a unit quaternion (shortest rotation)."""
    w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return np.array([2 * x, 2 * y, 2 * z])
    th = 2.0 * math.atan2(n, w)
    return np.array([x, y, z]) * (th / n)


def euler_zyx_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def tilt_angle(q) -> float:
    """Angle between body z and world z (rad)."""
    w, x, y, z = q
    c = 1 - 2 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, c)))


def hat(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def cross3(a, b) -> np.ndarray:
    """Cross product of two 3-vectors (much faster than ``np.cross`` for single vectors)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def norm3(a) -> float:
    return math.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2)


def np_clip(v, lo, hi) -> np.ndarray:
    """Element-wise clip; lower overhead than ``np.clip`` for tiny arrays."""
    return np.minimum(np.maximum(v, lo), hi)
