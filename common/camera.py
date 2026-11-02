"""카메라 내부 파라미터 (configs/camera_intrinsic.yaml): K 행렬, 왜곡 계수, Segments.ai 업로드 형식."""
import numpy as np

from .config import INTRINSICS_PATH
from .yaml_io import load_yaml


def load_intrinsics(path=INTRINSICS_PATH):
    """→ {카메라: {width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3}}"""
    return load_yaml(path)


def build_K(I):
    return np.array([[I["fx"], 0.0, I["cx"]], [0.0, I["fy"], I["cy"]], [0.0, 0.0, 1.0]])


def build_dist(I):
    return np.array([I["k1"], I["k2"], I["p1"], I["p2"], I["k3"]], np.float64)


def intrinsic_matrix_list(I):
    """Segments.ai 업로드용 K (리스트)."""
    return [[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]]


def distortion_dict(I):
    """Segments.ai 업로드용 왜곡 (brown-conrady)."""
    return {"model": "brown-conrady",
            "coefficients": {"k1": I["k1"], "k2": I["k2"], "k3": I["k3"], "p1": I["p1"], "p2": I["p2"]}}
