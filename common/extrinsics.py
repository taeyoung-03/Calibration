"""외부 파라미터 (configs/camera_lidar_extrinsic.yaml, LiDAR→카메라):
yaw/pitch/roll·tx/ty/tz ↔ (R, t), 프레임별 보정 회전, Segments.ai 업로드 형식(카메라→LiDAR + 쿼터니언)."""
import numpy as np
from scipy.spatial.transform import Rotation

from .config import EXTRINSICS_PATH
from .yaml_io import load_yaml

KEYS = ("yaw", "pitch", "roll", "tx", "ty", "tz")


def raw_to_Rt(e):
    """yaw/pitch/roll·tx/ty/tz → (R, t)."""
    return Rotation.from_euler("ZYX", [e["yaw"], e["pitch"], e["roll"]]).as_matrix(), np.array([e["tx"], e["ty"], e["tz"]])


def Rt_to_raw(R, t):
    y, p, r = Rotation.from_matrix(R).as_euler("ZYX")
    return {k: float(v) for k, v in zip(KEYS, (y, p, r, *np.ravel(t)))}


def load_extrinsics(path=EXTRINSICS_PATH):
    """→ (기준 {카메라: yaw…tz}, 기준 프레임 {카메라: 번호}, 프레임별 {"000123": {카메라: yaw…tz}})"""
    E = load_yaml(path)
    base = {c: e for c, e in E.items() if isinstance(e, dict) and "yaw" in e}
    return base, {c: int(n) for c, n in (E.get("reference") or {}).items()}, E.get("frames") or {}


def delta_R(I, du, dv):
    """픽셀 보정 (Δu, Δv) → 작은 회전 ΔR = Rx(−atan(Δv/fy))·Ry(atan(Δu/fx)) (카메라 좌표). I = 그 카메라 내부 파라미터."""
    return (Rotation.from_rotvec([-np.arctan(dv / I["fy"]), 0, 0]).as_matrix()
            @ Rotation.from_rotvec([0, np.arctan(du / I["fx"]), 0]).as_matrix())


def corrected_extrinsic(I, R, t, du, dv):
    """기준 외부 파라미터 (R, t)에 ΔR을 곱한 프레임별 외부 파라미터 (R′ = ΔR·R, t′ = ΔR·t). yaml과 같은 yaw…tz 형식."""
    dR = delta_R(I, du, dv)
    return Rt_to_raw(dR @ R, dR @ t)


def segments_extrinsics(e):
    """LiDAR→카메라 (yaw…tz) → Segments.ai 형식 (카메라→LiDAR 역변환 + 쿼터니언 [qx, qy, qz, qw])."""
    R, t = raw_to_Rt(e)
    R_inv, t_inv = R.T, -R.T @ t
    q = Rotation.from_matrix(R_inv).as_quat()
    return {"translation": {"x": float(t_inv[0]), "y": float(t_inv[1]), "z": float(t_inv[2])},
            "rotation": {"qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]), "qw": float(q[3])}}
