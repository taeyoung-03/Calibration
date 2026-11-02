"""투영·그리기: LiDAR 포인트 → 영상 픽셀, 보정 회전 후 다시 투영, 깊이 색 점 찍기, 블러, 양선형 샘플링."""
import cv2
import numpy as np

from .camera import build_dist, build_K
from .extrinsics import delta_R


def project(I, Rt, xyz):
    """LiDAR 포인트 → (픽셀 uv, 카메라 깊이 z, 원래 인덱스). 카메라 앞 0.5 m보다 먼 포인트 중 영상 안에 들어온 것만.
    I = 내부 파라미터, Rt = (R, t) LiDAR→카메라."""
    R, t = Rt
    pc = xyz @ R.T + t
    idx = np.where(pc[:, 2] > 0.5)[0]
    if not len(idx):
        return np.zeros((0, 2)), np.zeros(0), idx
    uv = cv2.projectPoints(pc[idx], np.zeros(3), np.zeros(3), build_K(I), build_dist(I))[0].reshape(-1, 2)
    m = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < I["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < I["height"])
    return uv[m], pc[idx[m], 2], idx[m]


def rotator(I, pc):
    """카메라 좌표 포인트 pc → place(Δu, Δv): ΔR로 돌린 뒤 내부 파라미터·왜곡으로 다시 투영한 픽셀 좌표."""
    K, dist = build_K(I), build_dist(I)

    def place(du, dv):
        if not len(pc):                                         # 포인트 없음 (projectPoints는 빈 배열을 못 받음)
            return np.zeros((0, 2))
        return cv2.projectPoints(pc, cv2.Rodrigues(delta_R(I, du, dv))[0], np.zeros(3), K, dist)[0].reshape(-1, 2)
    return place


def draw(img, uv, z, size):
    """포인트를 반지름 size의 채운 원으로 찍는다. 색 = 카메라 깊이 (JET, 가까울수록 파랑)."""
    out = img.copy()
    h, w = img.shape[:2]
    uv = uv.astype(int)
    m = (uv[:, 0] >= size) & (uv[:, 0] < w - size) & (uv[:, 1] >= size) & (uv[:, 1] < h - size)
    uv, z = uv[m], z[m]
    if len(z):
        col = cv2.applyColorMap(((z - z.min()) / (np.ptp(z) + 1e-6) * 255).astype(np.uint8),
                                cv2.COLORMAP_JET).reshape(-1, 3)
        stamp = cv2.circle(np.zeros((2 * size + 1, 2 * size + 1), np.uint8), (size, size), size, 1, -1)
        for dy, dx in np.argwhere(stamp) - size:
            out[uv[:, 1] + dy, uv[:, 0] + dx] = col
    return out


def blur(img, sigma):
    """가우시안 블러. sigma ≤ 0이면 그대로."""
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma and sigma > 0 else img


def sample_bilinear(img, u, v):
    h, w = img.shape
    x0, y0 = np.clip(u.astype(np.int64), 0, w - 2), np.clip(v.astype(np.int64), 0, h - 2)
    fx, fy = u - x0, v - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
