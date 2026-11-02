#!/usr/bin/env python3
"""LiDAR–카메라 정합 통합 실행.

대응점으로 카메라별 기준 외부 파라미터를 구하고(solvePnP), 프레임마다 투영한 포인트 클라우드 전체를 (Δu, Δv)만큼 옮겨
영상과 가장 잘 맞는 위치를 측정한 뒤, 시간 방향으로 정리하여(칼만 필터 + RTS smoother) 프레임·카메라별 외부 파라미터를 만든다.
보정 (Δu, Δv) px는 작은 회전 ΔR = Rx(−atan(Δv/fy))·Ry(atan(Δu/fx))로 바꿔 R′ = ΔR·R, t′ = ΔR·t로 적용한다.

단계 (기본은 전부 순서대로, --steps로 일부만 실행):
  pnp      configs/calibration_points.yaml의 2D–3D 대응점 → solvePnPRansac → 카메라별 기준 외부 파라미터
           → configs/camera_lidar_extrinsic.yaml (카메라 키, reference), out/calib/<카메라>_<프레임>.jpg (대응점 재투영 확인)
  measure  프레임·카메라마다 (Δu, Δv) 측정 → out/measure.csv
           Δv: LiDAR 반사 강도 ↔ 영상 밝기의 NID. Δu: 그 Δv의 가로선에서 깊이 맵(depth_map.py)과 LiDAR 거리의 분명한 모순 세기
  kalman   측정 → 고립·튐 제거 → 칼만 + RTS smoother → Δu 빈 구간은 다른 카메라 참고
           → out/kalman.csv, configs/camera_lidar_extrinsic.yaml의 frames (프레임·카메라별 외부 파라미터)
  render   프레임별 외부 파라미터로 포인트 클라우드를 투영한 확인 영상 → out/proj/<카메라>/<프레임>.jpg

사용법:
    python3 calibration.py                                   # 전체 (pnp → measure → kalman → render)
    python3 calibration.py --steps kalman render             # 측정은 두고 시간 방향 처리부터 다시
    python3 calibration.py --steps pnp --cam camera1 --img 2401 --pcd 2401   # reference가 빈 카메라의 기준 프레임 지정
설정: configs/calibration_config.yaml. measure 전에 depth_map.py로 깊이 맵을 만들어 둘 것.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # 프레임 병렬 실행이라 BLAS는 1스레드가 빠름
    os.environ.setdefault(_v, "1")

import argparse                                                             # noqa: E402
import csv                                                                  # noqa: E402
import multiprocessing                                                      # noqa: E402
import time                                                                 # noqa: E402

import cv2                                                                  # noqa: E402
import numpy as np                                                          # noqa: E402
from scipy.spatial import cKDTree                                           # noqa: E402

from common.camera import build_dist, build_K, load_intrinsics             # noqa: E402
from common.config import (EXTRINSICS_PATH, OUT, data_file, depth_path, frame_list, frame_number, img_path,  # noqa: E402
                           load_cfg, pcd_path)
from common.extrinsics import Rt_to_raw, corrected_extrinsic, load_extrinsics, raw_to_Rt  # noqa: E402
from common.pcd_io import read_pcd                                          # noqa: E402
from common.points import load_points                                       # noqa: E402
from common.projection import blur, draw, project, rotator, sample_bilinear  # noqa: E402
from common.yaml_io import update_yaml_entry                                # noqa: E402

STEPS = ("pnp", "measure", "kalman", "render")
MEASURE_CSV, KALMAN_CSV = "measure.csv", "kalman.csv"
# 작업 프로세스는 새로 띄움 (spawn). fork면 앞 단계(pnp)에서 쓴 OpenCV 스레드 상태가 복제되어 작업 프로세스가 멈춤
Pool = multiprocessing.get_context("spawn").Pool


def load_ctx():
    """내부 파라미터, 기준 외부 파라미터 {카메라: (R, t)}, 기준 프레임 {카메라: 번호}. 단계마다 새로 읽음 (pnp가 바꿨을 수 있어서)."""
    base, ref, _ = load_extrinsics()
    return {"intr": load_intrinsics(), "base": {c: raw_to_Rt(e) for c, e in base.items()}, "ref": ref}


def read_pcd_cfg(cfg, path):
    return read_pcd(path, cfg["lidar"]["field"], cfg["lidar"]["min_range"])


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])


def read_csv(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} 없음 → 앞 단계를 먼저 실행하세요")
    with open(path) as f:
        return list(csv.DictReader(f))


# =====================================================================================
# 1. pnp: 대응점 → 기준 외부 파라미터
# =====================================================================================
def match_pairs(imgs, pcds, mode):
    """--img·--pcd 짝짓기. order: i번째끼리, number: 파일 이름의 번호가 같은 것끼리."""
    if mode == "number":
        by_num = {frame_number(p): p for p in pcds}
        missing = [i for i in imgs if frame_number(i) not in by_num]
        if missing:
            raise SystemExit(f"--match number: 같은 번호의 포인트 클라우드 데이터가 없는 영상 {missing}")
        return [(i, by_num[frame_number(i)]) for i in imgs]
    if len(imgs) != len(pcds):
        raise SystemExit(f"--img {len(imgs)}개와 --pcd {len(pcds)}개의 개수가 다릅니다")
    return list(zip(imgs, pcds))


def solve_pnp(I, pts):
    """solvePnPRansac (RANSAC 가설 EPnP → inlier로 LM 정밀화). → (R, t, 영상 대응점, 재투영 위치)"""
    obj = np.array(pts.get("lidar_pts") or [], np.float64)
    img = np.array(pts.get("image_pts") or [], np.float64)
    if len(obj) != len(img) or len(obj) < 4:
        raise SystemExit(f"대응점 개수 문제 (lidar_pts {len(obj)}, image_pts {len(img)}, 같은 개수로 최소 4)")
    K, dist = build_K(I), build_dist(I)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
                                            reprojectionError=8.0, confidence=0.99)
    if not ok:
        raise SystemExit("solvePnP 실패")
    proj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    err = np.linalg.norm(proj - img, axis=1)
    print(f"  대응점 {len(obj)}개 (inlier {len(inl) if inl is not None else len(obj)}), "
          f"재투영 오차 평균 {err.mean():.2f}px / 최대 {err.max():.2f}px")
    for i, e in enumerate(err, 1):
        if e > 15:
            print(f"    ⚠️ {i}번째 대응점 재투영 오차 {e:.1f}px — 잘못 고른 점인지 확인")
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), img, proj


def step_pnp(cfg, args):
    pts = load_points()
    cams = args.cam or cfg["cameras"]
    ref = {c: dict((pts.get("reference") or {}).get(c) or {}) for c in cams}
    empty = [c for c in cams if not (ref[c].get("img") and ref[c].get("pcd"))]
    pairs = match_pairs(args.img, args.pcd, args.match) if (args.img or args.pcd) else []
    if len(pairs) > len(empty):
        raise SystemExit(f"--img/--pcd 쌍이 {len(pairs)}개인데 reference가 빈 카메라는 {empty} {len(empty)}대뿐입니다")
    for c, (i, p) in zip(empty, pairs):
        ref[c] = {"img": i, "pcd": p}
    skip = [c for c in cams if not (ref[c].get("img") and ref[c].get("pcd"))]
    if skip:
        print(f"reference가 비어 건너뜀 (기존 외부 파라미터 유지): {skip}")

    intr = load_intrinsics()
    _, reference, _ = load_extrinsics()
    os.makedirs(os.path.join(OUT, "calib"), exist_ok=True)
    for cam in (c for c in cams if c not in skip):
        ip = data_file(cfg, ref[cam]["img"], os.path.join("CAMERA", cam), ".jpg")
        pp = data_file(cfg, ref[cam]["pcd"], "LiDAR", ".pcd")
        n_img, n_pcd = frame_number(ip), frame_number(pp)
        print(f"[{cam}] 기준 프레임 {n_img:06d}: {ip} + {pp}")
        if n_img != n_pcd:
            print(f"  ⚠️ 영상({n_img})과 포인트 클라우드 데이터({n_pcd}) 번호가 다름. 기준 프레임은 영상 번호로 둠")
        if cam not in pts:
            raise SystemExit(f"{cam}: calibration_points.yaml에 대응점이 없음")
        R, t, img_pts, proj = solve_pnp(intr[cam], pts[cam])
        update_yaml_entry(EXTRINSICS_PATH, cam, Rt_to_raw(R, t))
        reference[cam] = n_img

        img = cv2.imread(ip)
        if img is not None and os.path.exists(pp):
            xyz, _ = read_pcd_cfg(cfg, pp)
            uv, z, _ = project(intr[cam], (R, t), xyz)
            out = draw(img, uv, z, cfg["render"]["point_size"])
            for (u, v) in img_pts.astype(int):
                cv2.circle(out, (u, v), 8, (0, 255, 0), 2)                         # 초록: 고른 영상 대응점
            for (u, v) in proj.astype(int):
                cv2.drawMarker(out, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)  # 빨강: 3D 대응점 재투영
            cv2.imwrite(os.path.join(OUT, "calib", f"{cam}_{n_img:06d}.jpg"), out)

    update_yaml_entry(EXTRINSICS_PATH, "reference", {c: int(reference[c]) for c in sorted(reference)})
    print(f"기준 외부 파라미터·기준 프레임 → {EXTRINSICS_PATH}, 확인 영상 → {os.path.join(OUT, 'calib')}")


# =====================================================================================
# 2. measure: 프레임·카메라별 (Δu, Δv)
# =====================================================================================
def normalize(v):
    """LiDAR 반사 강도: log1p 후 1~99% 구간을 [0, 1]로."""
    v = np.log1p(np.maximum(v, 0))
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)


# ---- Δv: 밝기 NID ----
def _entropy(p):
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def nid_cost(x, y, bins):
    """x, y: [0,1] 값. 선형 보간 히스토그램(soft binning)으로 NID = (2H(X,Y) − H(X) − H(Y)) / H(X,Y). 낮을수록 잘 맞음."""
    B = bins
    px, py = x * (B - 1), y * (B - 1)
    ix, iy = np.minimum(px.astype(np.int64), B - 2), np.minimum(py.astype(np.int64), B - 2)
    fx, fy = px - ix, py - iy
    H = np.zeros(B * B)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            H += np.bincount((ix + dx) * B + (iy + dy), weights=wx * wy, minlength=B * B)
    P = H.reshape(B, B) / len(x)
    hxy = _entropy(P.ravel())
    if hxy <= 0:
        return 1.0
    return (2 * hxy - _entropy(P.sum(1)) - _entropy(P.sum(0))) / hxy


def find(cost, pos, coarse, N):
    """거친 전수 탐색 → 거친 최저점 주변 ±fine_px를 1px 간격으로 → (Δu, Δv, 점수, 거친 점수들, 거친 최저점)."""
    sc = np.array([cost(*pos(du, dv).T) for du, dv in coarse])
    bu, bv = coarse[int(np.argmin(sc))]
    f = range(-N["fine_px"], N["fine_px"] + 1)
    fine = [(bu + du, bv + dv) for dv in f for du in f]
    sf = np.array([cost(*pos(du, dv).T) for du, dv in fine])
    du, dv = fine[int(np.argmin(sf))]
    return du, dv, float(sf.min()), sc, (bu, bv)


def on_edge(N, key, x):
    """답이 탐색 범위 끝 (edge_margin_px 안) = 범위 안에서 최저점을 못 찾음."""
    return abs(x) >= N["search_px"][key] - N["edge_margin_px"]


def v_check(sc, coarse, peak, du, dv, best, N):
    """Δv 판정: 거친 최저점에서 sep_px 밖 거친 점수의 최저와 답 점수의 차 = 뚜렷함. → (뚜렷함, 사용 여부)"""
    far = np.hypot(*(np.array(coarse) - peak).T) > N["hist"]["sep_px"]
    dist = float(sc[far].min() - best)
    return dist, int(dist >= N["hist"]["min_distinct"] and not on_edge(N, "u", du) and not on_edge(N, "v", dv))


# ---- Δu: 깊이 맵과 LiDAR 거리의 분명한 모순 세기 ----
def neighbor_pairs(uv, z, P):
    """기준 위치 영상에서 포인트마다 가까운 neighbors개 (max_px 안) 이웃과 쌍 (방향 무관). 같은 쌍은 한 번만.
    → (i, j, LiDAR 깊이 변화 log z_j − log z_i)"""
    k = min(P["neighbors"] + 1, len(uv))
    d, nb = cKDTree(uv).query(uv, k=k, distance_upper_bound=P["max_px"])
    i, j = np.repeat(np.arange(len(uv)), k - 1), nb[:, 1:].ravel()
    ok = np.isfinite(d[:, 1:].ravel()) & (i != j)
    i, j = np.unique(np.sort(np.c_[i[ok], j[ok]], 1), axis=0).T
    return i, j, np.log(z[j]) - np.log(z[i])


def pair_groups(i, j, z, dl, O):
    """이웃 쌍 분류. r = +1 (i가 j보다 edge_ratio 배 넘게 가까움) / −1 (멂) / 0 (이어짐), e = 가까움·멂 쌍 여부.
    거리대 = 쌍의 가까운 포인트 거리를 group_ratio 배 간격으로 나눈 칸 (깊이 맵 흔들림을 거리대마다 따로 잼)."""
    r = np.sign(dl) * (np.abs(dl) > np.log(O["edge_ratio"]))     # dl = log z_j − log z_i > 0 → i가 가까움 → +1
    zn = np.minimum(z[i], z[j])
    g = np.floor(np.log(zn / zn.min()) / np.log(O["group_ratio"])).astype(np.int64)
    e = r != 0
    return {"i": i, "j": j, "r": r, "e": e, "ge": g[e], "gc": g[~e], "G": g.max() + 1}


def depth_noise(depth, pos, Q, line):
    """거리대마다 깊이 맵 흔들림 수준 (프레임마다 하나): LiDAR로는 이어진 쌍의 깊이 맵 차이 |ΔD| 중앙값을
    가로선 후보마다 구해 다시 중앙값. 가까움·멂 쌍의 방향은 이보다 작은 차이로는 반대로 판정하지 않음."""
    c = ~Q["e"]
    i, j = Q["i"][c], Q["j"][c]
    M = []
    for du, dv in line:
        D = sample_bilinear(depth, *pos(du, dv).T)
        a = np.abs(D[i] - D[j])
        M.append([np.median(a[Q["gc"] == g]) if np.any(Q["gc"] == g) else 0.0 for g in range(Q["G"])])
    return np.median(np.array(M), axis=0)


def sky_core(d_raw, O):
    """확실한 하늘: 깊이 모델이 하늘로 칠한 곳(블러 전 값 ≤ sky_thr)이면서 화면 맨 위와 이어진 영역에서, 경계 sky_edge_px 안쪽.
    깊이 맵 경계 번짐·LiDAR 빔 퍼짐으로 맞는 위치에서도 물체 테두리 포인트가 하늘 경계에 살짝 걸리는 것은 안 셈."""
    m = (d_raw <= O["sky_thr"]).astype(np.uint8)
    _, lab = cv2.connectedComponents(m, connectivity=4)
    top = np.unique(lab[0][m[0] > 0])
    sky = np.isin(lab, top[top > 0]).astype(np.uint8)
    k = 2 * O["sky_edge_px"] + 1
    return cv2.erode(sky, np.ones((k, k), np.uint8)).astype(bool)


def _pav_decreasing(y, wt):
    """가중 단조 감소 회귀 (인접 위반 합치기)."""
    out = []
    for v, ww in zip(y, wt):
        out.append([v, ww, 1])
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            v2, w2, n2 = out.pop()
            v1, w1, n1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[v] * c for v, _, c in out])


def _quantile_bins(lz, nb):
    """log 거리를 분위 nb칸으로 나눈 칸별 포인트 번호 (빈 칸 제외)."""
    q = np.quantile(lz, np.linspace(0, 1, nb + 1))
    q[-1] += 1e-9
    k = np.clip(np.searchsorted(q, lz, "right") - 1, 0, nb - 1)
    return [b for b in (np.where(k == g)[0] for g in range(nb)) if len(b)]


def depth_fit(depth, pos, lz, Q, line, O):
    """분포 기준 (프레임마다 하나. 가로선 후보들의 중앙값으로 고정 — 후보마다 다시 맞추면 틀린 위치에도 맞춰짐):
      f(z): LiDAR 거리 → 정상 깊이 맵 값. 후보마다 log 거리 분위 curve_bins칸의 깊이 맵 중앙값을 단조 감소 회귀
            (멀수록 작아진다는 것만 가정. 직선(a/z + b)이나 로그 같은 모양은 정하지 않음)
      σ(z): 그 거리의 정상 흔들림 = 분위 noise_bins칸마다 (깊이 맵 − f)의 MAD. 하한 = 깊이 맵 국소 흔들림 (이어짐 쌍 |ΔD| 중앙값)
    → 포인트마다 (f, σ)"""
    D = [sample_bilinear(depth, *pos(du, dv).T) for du, dv in line]
    bins = _quantile_bins(lz, O["curve_bins"])
    c, cnt = np.array([np.median(lz[b]) for b in bins]), np.array([len(b) for b in bins])
    grid = np.linspace(lz.min(), lz.max(), 80)
    F = np.median([np.interp(grid, c, _pav_decreasing([np.median(d[b]) for b in bins], cnt)) for d in D], axis=0)
    f = np.maximum(np.interp(lz, grid, F), 0)
    bins = [b for b in _quantile_bins(lz, O["noise_bins"]) if len(b) > 1]
    mad = lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))
    sb = np.median([[mad(d[b] - f[b]) for b in bins] for d in D], axis=0)
    cont = ~Q["e"]
    floor = np.median([np.median(np.abs(d[Q["i"][cont]] - d[Q["j"][cont]])) for d in D]) if cont.any() else 0.0
    sig = np.maximum(np.interp(lz, [np.median(lz[b]) for b in bins], sb), max(floor, 1e-6))
    return f, sig


def contradiction_curves(depth, core, pos, us, v0, Q, O, f, sig, surf):
    """후보 Δu마다 (하늘 포인트 수 S, 경계 모순 쌍 수 V, 분포 점수 C). 깊이 맵이 LiDAR와 분명히 모순되는 것만 크게 셈.
      S: 확실한 하늘(sky_core)에 떨어진 포인트 중 surf (LiDAR로 이어진 이웃이 있는 표면 포인트. 새·물보라처럼 혼자 떨어진 포인트는 빠짐)
      V: 이어짐 쌍인데 깊이 맵이 break_ratio 배 넘게 끊김 + 가까움·멂 쌍인데 깊이 맵 방향이 판정 하한 넘게 반대. 모순 쌍 하나에 1
         판정 하한 s = max(sign_soft × 두 값 평균, noise_mult × 그 거리대 깊이 맵 흔들림)
      C: Σ_포인트 min(((깊이 맵 − f) / σ)², cap_sigma²). 정상 흔들림 안은 작게, cap_sigma배 넘게 벗어나면 모순 하나와 같은 값"""
    h, w = core.shape
    e, cap2, lb = Q["e"], O["cap_sigma"] ** 2, np.log(O["break_ratio"])
    S, V, C = [], [], []
    for du in us:
        u, v = pos(du, v0).T
        D = sample_bilinear(depth, u, v)
        on = core[np.clip(np.rint(v).astype(np.int64), 0, h - 1), np.clip(np.rint(u).astype(np.int64), 0, w - 1)]
        Di, Dj = D[Q["i"]], D[Q["j"]]
        s = np.maximum(O["sign_soft"] * (0.5 * (Di[e] + Dj[e]) + O["eps"]), O["noise_mult"] * Q["noise"][Q["ge"]])
        S.append(int((on & surf).sum()))
        V.append(int((Q["r"][e] * (Di[e] - Dj[e]) < -s).sum() + (np.abs(np.log((Di[~e] + O["eps"]) / (Dj[~e] + O["eps"]))) > lb).sum()))
        C.append(float(np.minimum(((D - f) / sig) ** 2, cap2).sum()))
    return np.array(S), np.array(V), np.array(C)


def pick_u(us, S, T, N):
    """Δu 고르기: 하늘 포인트가 가장 적은 위치보다 sky_tol·√(최저+1)개 넘게 많은 위치는 안 고름 (어느 위치에나 있는 잘못 찍힌
    포인트는 최저에 들어가 상쇄되고, √는 셈 자체의 흔들림). 남은 위치 중 T(= cap² × 경계 모순 + 분포 점수)가 가장 작은 곳.
    뚜렷함 = (거친 간격 위치 중 axis_sep_px 밖 최저 T − 답의 T) / 거친 간격 위치 T의 표준편차 (못 고르는 위치는 T 최댓값으로).
    → (Δu, 뚜렷함, 사용 여부)"""
    ok = S <= S.min() + N["depth_score"]["sky_tol"] * np.sqrt(S.min() + 1)
    Tm = np.where(ok, T, T.max())
    k = int(np.argmin(Tm))
    du, best = int(us[k]), float(Tm[k])
    p = us % N["coarse_px"] == 0
    far = p & (np.abs(us - du) > N["axis_sep_px"])
    spread = float(np.std(np.r_[Tm[p], best]))
    dist = float((Tm[far].min() - best) / spread) if far.any() and spread > 0 else 0.0
    return du, dist, int(dist >= N["min_distinct_rel"] and not on_edge(N, "u", du))


def measure(cfg, ctx, cam, n, xyz, val):
    """한 프레임·카메라의 (Δu, Δv) 측정 → measure.csv 한 줄 (영상·깊이 맵이 없으면 None)."""
    N, O = cfg["measure"], cfg["measure"]["depth_score"]
    I, (R, t) = ctx["intr"][cam], ctx["base"][cam]
    img = cv2.imread(img_path(cfg, cam, n), cv2.IMREAD_GRAYSCALE)
    d = cv2.imread(depth_path(cfg, cam, n), cv2.IMREAD_UNCHANGED)
    if img is None or d is None:                                # 손상된 JPG, 깊이 맵 없음
        return None
    gray = blur(img.astype(np.float32) / 255.0, N["blur_sigma"])
    d_raw = cv2.resize(d.astype(np.float32) / 65535.0, img.shape[::-1], interpolation=cv2.INTER_LINEAR)  # 깊이 맵을 원본 크기로
    dmap = blur(d_raw, N["blur_sigma"])
    h, w = gray.shape
    ru, rv = N["search_px"]["u"], N["search_px"]["v"]
    Su, Sv = ru + N["fine_px"] + 1, rv + N["fine_px"] + 1
    uv, z, idx = project(I, (R, t), xyz)
    vmax = h * (1 - N["mask_bottom"][cam])
    pc = xyz[idx] @ R.T + t
    corner = rotator(I, pc)
    keep = uv[:, 1] < vmax                                      # 탐색 범위 네 귀퉁이 회전에서도 영상 안에 남는 포인트만 (모든 후보에서 같은 포인트 집합)
    for du, dv in ((-Su, -Sv), (-Su, Sv), (Su, -Sv), (Su, Sv)):
        q = corner(du, dv)
        keep &= (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < vmax)
    uv, x, z = uv[keep], normalize(val[idx[keep]]) if keep.any() else val[:0], z[keep]
    if len(uv) < N["min_points"]:
        return [n, cam, len(uv), "", 0, "", "", 0, ""]
    pos = rotator(I, pc[keep])
    c = N["coarse_px"]
    coarse = [(du, dv) for dv in np.arange(-rv, rv + 1, c) for du in np.arange(-ru, ru + 1, c)]

    # Δv 먼저: 밝기 NID
    cost_v = lambda u, v: nid_cost(x, sample_bilinear(gray, u, v), N["hist"]["bins"])
    du_v, dv, best_v, sc, peak = find(cost_v, pos, coarse, N)
    dist_v, used_v = v_check(sc, coarse, peak, du_v, dv, best_v, N)

    # Δu: 위 Δv의 가로선에서만 찾음 (Δv 측정이 사용 판정을 못 받으면 Δv = 0).
    # Δu 점수가 Δv까지 움직이면 수평선·해안선처럼 가로로 긴 구조 때문에 엉뚱한 Δv에서 가짜 최저점이 생김
    v0 = int(dv) if used_v else 0
    line = [(du, v0) for du in np.arange(-ru, ru + 1, c)]
    si, sj, dl = neighbor_pairs(uv, z, N["pairs"])
    Q = pair_groups(si, sj, z, dl, O)
    Q["noise"] = depth_noise(dmap, pos, Q, line)
    us = np.arange(-ru, ru + 1)                                 # 1px 간격 전 범위
    f, sig = depth_fit(dmap, pos, np.log(z), Q, line, O)
    surf = np.ones(len(z), bool)
    if O["sky_skip_isolated"]:                                  # LiDAR로 이어진 이웃이 있는 포인트만 (혼자 떨어진 포인트는 하늘 포인트로 안 셈)
        surf[:] = False
        surf[Q["i"][~Q["e"]]] = surf[Q["j"][~Q["e"]]] = True
    core = sky_core(d_raw, O)
    S, V, C = contradiction_curves(dmap, core, pos, us, v0, Q, O, f, sig, surf)
    # 기준표 보완: 경계 모순이 최저 수준(최저 + √)인 위치들이 axis_sep_px 안에 모이면(경계가 위치를 분명히 정함),
    # 거리별 정상값·흔들림을 그 위치들에서만 다시 잼 (틀리게 옮긴 위치들의 값이 기준표를 끌고 가지 않게). 넓게 퍼지면 그대로
    near = us[V <= V.min() + np.sqrt(V.min() + 1)]
    if near.max() - near.min() <= N["axis_sep_px"]:
        f, sig = depth_fit(dmap, pos, np.log(z), Q, [(d, v0) for d in near], O)
        S, V, C = contradiction_curves(dmap, core, pos, us, v0, Q, O, f, sig, surf)
    du, dist_u, used_u = pick_u(us, S, O["cap_sigma"] ** 2 * V + C, N)
    return [n, cam, len(uv), du, used_u, f"{dist_u:.3f}", dv, used_v, f"{dist_v:.4f}"]


def measure_frame(job):
    cfg, ctx, n = job
    cv2.setNumThreads(1)
    xyz, val = read_pcd_cfg(cfg, pcd_path(cfg, n))
    return [r for cam in cfg["cameras"] if (r := measure(cfg, ctx, cam, n, xyz, val))]


def step_measure(cfg, args):
    ctx = load_ctx()
    fr = frame_list(cfg, ctx["ref"].values())
    ddir = os.path.join(OUT, cfg["depth"]["dir"])
    if not os.path.isdir(ddir):
        raise SystemExit(f"깊이 맵 폴더 {ddir} 없음 → depth_map.py를 먼저 실행하세요")
    rows = []
    with Pool(cfg["workers"]) as p:
        for i, part in enumerate(p.imap(measure_frame, [(cfg, ctx, n) for n in fr]), 1):
            rows += part
            if i % 50 == 0 or i == len(fr):
                print(f"{i}/{len(fr)} 프레임", flush=True)
    write_csv(MEASURE_CSV, ["frame", "camera", "n_points", "du", "used_u", "distinct_u", "dv", "used_v", "distinct_v"], rows)
    for cam in cfg["cameras"]:
        r = [x for x in rows if x[1] == cam]
        print(f"{cam}: 측정 사용 Δu {sum(x[4] for x in r)}, Δv {sum(x[7] for x in r)} / {len(r)} 프레임")
    print(f"→ {os.path.join(OUT, MEASURE_CSV)}")


# =====================================================================================
# 3. kalman: 시간 방향 처리 → 프레임별 외부 파라미터
# =====================================================================================
# [Δu, Δv 공통] 튐 제거 → 앞뒤 잇기 칼만 + RTS smoother: 측정 프레임은 측정값을 거의 그대로, 빈 프레임은 앞뒤 측정 사이를 이음
#   예측:  x⁻_k = a·x_{k−1},  P⁻_k = a²·P_{k−1} + q²   (q = 칼만 프레임 한 칸 사이 실제로 바뀌는 폭)
#          Δu: a = 1 (빈 구간은 앞뒤 측정 사이 직선)
#          Δv: a = exp(−1/v_decay_frames) (기준 외부 파라미터의 상하가 0이라 측정 없는 긴 구간은 천천히 0으로 돌아감)
#   갱신:  G = P⁻/(P⁻ + R_k),  x = x⁻ + G·(z − x⁻),  P = (1 − G)·P⁻
#          R_k = σ² · (카메라별 뚜렷함 중앙값 / 그 측정의 뚜렷함). 뚜렷한 측정일수록 측정값에 더 가깝게.
# [Δu 빈 구간] 다른 카메라 참고 (cross_fill): 배는 하나라 느린 어긋남 흐름은 카메라끼리 같은 방향.
# 기준 프레임(camera_lidar_extrinsic.yaml의 reference)은 보정 0으로 고정 (거기서 뒤쪽·앞쪽으로 각각 필터 + smoother). 없으면 첫 프레임.
def kalman_rts(z, r, q2, a):
    """앞뒤 잇기 칼만 + RTS smoother. z, r: 프레임별 측정·분산 (측정 없으면 NaN), q2: 예측 분산, a: 상태 유지 비율.
    첫 원소 = 기준 프레임 (보정 0·불확실성 0). → 평활 값, 평활 표준편차"""
    n = len(z)
    xp, Pp, xf, Pf = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    x, P = 0.0, 0.0
    for k in range(n):
        if k:
            x, P = a * x, a * a * P + q2                            # 예측
        xp[k], Pp[k] = x, P
        if k and z[k] == z[k]:                                      # 측정 있음 (NaN 아님)
            G = P / (P + r[k])
            x, P = x + G * (z[k] - x), (1 - G) * P
        xf[k], Pf[k] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):                                  # RTS smoother
        C = a * Pf[k] / Pp[k + 1]
        xs[k] = xf[k] + C * (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C * C * (Ps[k + 1] - Pp[k + 1])
    return xs, np.sqrt(np.maximum(Ps, 0))


def kalman_from_reference(z, r, q2, a, k0):
    """기준 프레임 인덱스 k0에서 보정 0으로 시작해 뒤쪽(k0→끝)과 앞쪽(k0→처음)으로 각각 필터 + smoother."""
    xs, ss = np.zeros(len(z)), np.zeros(len(z))
    xs[k0:], ss[k0:] = kalman_rts(z[k0:], r[k0:], q2, a)
    if k0 > 0:
        idx = np.arange(k0, -1, -1)                                   # 앞쪽: k0, k0−1, …, 0
        xs[idx], ss[idx] = kalman_rts(z[idx], r[idx], q2, a)
    return xs, ss


def change_q(G, fr, sigma, R, max_lag):
    """칼만 한 칸 사이 실제 변화 q (px)를 데이터에서. L칸 떨어진 측정 쌍의 차이 분산(MAD) s²(L) = 2σ² + L·q².
    max_lag = 1: q² = s²(1) − 2σ² (σ = 측정 잡음, 설정).
    max_lag > 1: L = 1..max_lag의 s²(L)를 L에 직선으로 맞춘 기울기 = q² (잡음이 커서 1칸 차이로는 느린 변화가 안 보일 때. Δv).
    쌍이 min_pairs보다 적으면 fallback_px, 하한 floor_px. → (q, 1칸 쌍 수)"""
    idx = {n: i for i, n in enumerate(fr)}
    ns = sorted(n for n in G if n in idx)
    mad2 = lambda d: (1.4826 * np.median(np.abs(d - np.median(d)))) ** 2
    lags, s2, n1 = [], [], 0
    for L in range(1, max_lag + 1):
        d = np.array([G[b] - G[a] for a in ns for b in ns if idx[b] - idx[a] == L])
        n1 = n1 or len(d)
        if len(d) >= R["min_pairs"]:
            lags.append(L)
            s2.append(mad2(d))
    if not lags or (max_lag > 1 and len(lags) < 2):
        return float(R["fallback_px"]), n1
    q2 = s2[0] - 2 * sigma ** 2 if max_lag == 1 else np.polyfit(lags, s2, 1)[0]
    return float(np.sqrt(max(q2, R["floor_px"] ** 2))), n1


def drop_spikes(G, fr, sigma, q, S):
    """튐 제거: 측정마다 바로 앞·뒤 측정(칼만 프레임 max_gap칸 안)을 보고, 앞뒤끼리는 허용 폭 안에서 맞는데
    이 측정이 둘 다에서 허용 폭 넘게 벗어나면 버림. 허용 폭 = gate·√(2σ² + 칸 수·q²) (두 측정 차이의 표준편차).
    한쪽과 이어지면 급변, 앞뒤끼리도 다르면 판단 근거 없음 → 유지. → (남은 측정, 버린 프레임)"""
    idx = {n: i for i, n in enumerate(fr)}
    ns = sorted(n for n in G if n in idx)
    ok = lambda a, b: abs(G[b] - G[a]) <= S["gate"] * np.sqrt(2 * sigma ** 2 + (idx[b] - idx[a]) * q ** 2)
    near = lambda a, b: idx[b] - idx[a] <= S["max_gap"]
    bad = [b for a, b, c in zip(ns, ns[1:], ns[2:]) if near(a, b) and near(b, c) and ok(a, c) and not ok(a, b) and not ok(b, c)]
    return {n: x for n, x in G.items() if n not in bad}, bad


def drop_isolated(m, win, tol):
    """{프레임: 측정값} 중 프레임 번호로 앞뒤 win 안에 tol 이내로 비슷한 다른 측정이 없는 것을 뺀다."""
    return {n: x for n, x in m.items() if any(abs(k - n) <= win and abs(y - x) <= tol for k, y in m.items() if k != n)}


def track(G, fr, distinct, sigma, a, k0, K, max_lag):
    """한 카메라·한 축: 튐 제거 → 칼만. distinct: {프레임: 뚜렷함}. → (값, 표준편차, 튐 여부, 남은 측정, 로그용 (q, 쌍 수, 버린 프레임, 뚜렷함 중앙값))"""
    q, npair = change_q(G, fr, sigma, K["r_data"], max_lag)
    G, bad = drop_spikes(G, fr, sigma, q, K["spike"])
    ref_d = float(np.median([distinct[n] for n in G])) if G else 1.0
    z, r = np.full(len(fr), np.nan), np.full(len(fr), np.nan)
    for i, n in enumerate(fr):
        if n in G:
            z[i], r[i] = G[n], sigma ** 2 * ref_d / max(distinct[n], 1e-6)
    xs, ss = kalman_from_reference(z, r, q * q, a, k0)
    return xs, ss, np.array([n in bad for n in fr]), G, (q, npair, bad, ref_d)


def cross_fill(xs, fr, cam, kept, k0, C, intr):
    """[Δu 빈 구간] 다른 카메라 참고. kept: {카메라: 튐·고립 뺀 Δu 측정}. 다른 카메라 측정을 각자 편향(중앙값)을 빼고
    각도(÷fx)로 바꿔 프레임마다 평균 = 공통 흐름. 이 카메라의 양 끝이 측정인 빈 구간(min_frames칸 이상)마다, 공통 흐름이
    구간 양 끝을 이은 직선에서 위나 아래로 벗어났다 돌아오면 (방향이 바뀜) 벗어난 만큼 × β를 이 카메라 px로 더함 (β·벗어난 폭이 tol 넘을 때만).
    β = 같은 프레임에 둘 다 측정이 있을 때 이 카메라 편향 뺀 값을 공통 흐름에 회귀한 기울기 (0~1).
    → (값, 바꾼 프레임 여부, β)"""
    fx = intr[cam]["fx"]
    acc = np.zeros(len(fr)); cnt = np.zeros(len(fr))
    for o, G in kept.items():
        if o == cam or not G:
            continue
        b = np.median(list(G.values()))
        for i, n in enumerate(fr):
            if n in G:
                acc[i] += (G[n] - b) / intr[o]["fx"]
                cnt[i] += 1
    common = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan) * fx           # 공통 흐름 (이 카메라 px)
    own = np.array([kept[cam].get(n, np.nan) for n in fr]) - (np.median(list(kept[cam].values())) if kept[cam] else 0.0)
    both = ~np.isnan(own) & ~np.isnan(common)
    beta = 0.0                                                      # 이 카메라가 공통 흐름을 얼마나 따르나 (회귀 기울기, 0~1)
    if both.sum() >= 10 and common[both].var() > 0:
        beta = float(np.clip(np.cov(common[both], own[both])[0, 1] / common[both].var(), 0, 1))
    meas = np.array([n in kept[cam] for n in fr])
    meas[k0] = True
    out, moved = xs.copy(), np.zeros(len(fr), bool)
    m_idx = np.flatnonzero(meas)
    for A, B in zip(m_idx, m_idx[1:]):
        if B - A - 1 < C["min_frames"]:
            continue
        seg = np.arange(A, B + 1)
        av = seg[~np.isnan(common[seg])]
        if len(av) < 3:
            continue
        c = np.interp(seg, av, common[av])                          # 빈 칸은 다른 카메라끼리도 직선으로
        chord = c[0] + (c[-1] - c[0]) * (seg - A) / (B - A)
        bump = beta * (c - chord)
        if bump.max() > C["tol_px"] or bump.min() < -C["tol_px"]:
            out[A + 1:B] += bump[1:-1]
            moved[A + 1:B] = True
    return out, moved, beta


def step_kalman(cfg, args):
    ctx = load_ctx()
    K, fr = cfg["kalman"], frame_list(cfg, ctx["ref"].values())
    meas = {(int(r["frame"]), r["camera"]): r for r in read_csv(MEASURE_CSV)}
    win, tol = K["isolation"]["window_frames"], K["isolation"]["tol_px"]
    a_v = float(np.exp(-1.0 / K["v_decay_frames"]))
    res, kept, k0s = {}, {}, {}
    for cam in cfg["cameras"]:
        pick = lambda ax: {n: float(m[ax]) for n in fr if (m := meas.get((n, cam))) and m[f"used_{ax[1]}"] == "1"}
        raw_u, raw_v = pick("du"), pick("dv")
        good = (drop_isolated(raw_u, win, tol), drop_isolated(raw_v, win, tol))
        print(f"{cam}: 측정 Δu {len(raw_u)} → 고립 제외 {len(good[0])}, Δv {len(raw_v)} → 고립 제외 {len(good[1])}")
        ref = ctx["ref"].get(cam)
        k0s[cam] = k0 = fr.index(ref) if ref in fr else 0
        print(f"{cam}: 기준 프레임 {fr[k0]:06d}" + ("" if ref in fr else " (reference 없음 또는 범위 밖 → 첫 프레임)"))
        res[cam] = []
        for G, ax, sigma, a, L in ((good[0], "u", K["u_sigma_px"], 1.0, 1), (good[1], "v", K["v_sigma_px"], a_v, K["r_data"]["v_max_lag"])):
            distinct = {n: float(meas[(n, cam)][f"distinct_{ax}"]) for n in G}
            xs, ss, sp, Gk, (q, npair, bad, ref_d) = track(G, fr, distinct, sigma, a, k0, K, L)
            print(f"   d{ax}: 측정 잡음 σ {sigma:.2f}px, 한 칸 실제 변화 q {q:.2f}px (연속 측정 쌍 {npair}), "
                  f"뚜렷함 중앙값 {ref_d:.4g}, 튐 제거 {len(bad)}개 {bad}")
            res[cam].append([xs, ss, sp])
            if ax == "u":
                kept[cam] = Gk
    rows = []
    for cam in cfg["cameras"]:
        res[cam][0][0], moved, beta = cross_fill(res[cam][0][0], fr, cam, kept, k0s[cam], K["cross"], ctx["intr"])
        print(f"{cam}: Δu 빈 구간 다른 카메라 참고 β {beta:.2f}, 바꾼 프레임 {int(moved.sum())}개")
        (xu, su, pu), (xv, sv, pv) = res[cam]
        for i, n in enumerate(fr):
            rows.append([n, cam, f"{xu[i]:.1f}", f"{xv[i]:.1f}", f"{su[i]:.1f}", f"{sv[i]:.1f}", int(pu[i]), int(pv[i]), int(moved[i])])
    write_csv(KALMAN_CSV, ["frame", "camera", "kf_du", "kf_dv", "sigma_u", "sigma_v", "spike_u", "spike_v", "cross_u"], rows)

    # 프레임·카메라별 외부 파라미터를 camera_lidar_extrinsic.yaml의 frames 항목에 기록 (기준 항목은 그대로)
    ext = {}
    for n, cam, du, dv, *_ in rows:
        ext.setdefault(f"{n:06d}", {})[cam] = corrected_extrinsic(ctx["intr"][cam], *ctx["base"][cam], float(du), float(dv))
    update_yaml_entry(EXTRINSICS_PATH, "frames", ext)
    print(f"→ {os.path.join(OUT, KALMAN_CSV)}, 프레임별 외부 파라미터 {len(ext)}프레임 → {EXTRINSICS_PATH} (frames)")


# =====================================================================================
# 4. render: 확인용 투영 영상
# =====================================================================================
def render_frame(job):
    """프레임 하나의 카메라별 투영 영상. kf = {카메라: kalman.csv 행}"""
    cfg, ctx, kf, n = job
    cv2.setNumThreads(1)
    xyz, _ = read_pcd_cfg(cfg, pcd_path(cfg, n))
    for cam in cfg["cameras"]:
        img = cv2.imread(img_path(cfg, cam, n))
        k = kf.get(cam)
        if img is None or k is None:
            continue
        du, dv = float(k["kf_du"]), float(k["kf_dv"])
        I = ctx["intr"][cam]
        uv, z, _ = project(I, raw_to_Rt(corrected_extrinsic(I, *ctx["base"][cam], du, dv)), xyz)
        out = draw(img, uv, z, cfg["render"]["point_size"])
        flags = "".join(f" {name}" for name, col in (("SPIKE_U", "spike_u"), ("SPIKE_V", "spike_v"), ("CROSS_U", "cross_u")) if k.get(col) == "1")
        cv2.putText(out, f"{cam} {n:06d} du={du:+.1f} dv={dv:+.1f} sigma=({k['sigma_u']},{k['sigma_v']}){flags}",
                    (10, 30), 0, 1.2, (255, 255, 255), 2)
        d = os.path.join(OUT, "proj", cam)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{n:06d}.jpg"), out)


def step_render(cfg, args):
    ctx = load_ctx()
    f = cfg["frames"]
    kf = {}
    for r in read_csv(KALMAN_CSV):
        kf.setdefault(int(r["frame"]), {})[r["camera"]] = r
    jobs = [(cfg, ctx, kf.get(n, {}), n) for n in range(f["start"], f["end"] + 1, cfg["render"]["step"])]
    with Pool(cfg["workers"]) as p:
        for i, _ in enumerate(p.imap_unordered(render_frame, jobs), 1):
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} 프레임", flush=True)
    print("→", os.path.join(OUT, "proj"))


# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description="LiDAR–카메라 정합: 기준 외부 파라미터 → 프레임별 측정 → 시간 방향 처리 → 확인 영상")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="실행할 단계 (기본: 전부, 항상 pnp→measure→kalman→render 순)")
    ap.add_argument("--cam", nargs="+", default=None, help="pnp: 처리할 카메라 (기본: 설정의 cameras)")
    ap.add_argument("--img", nargs="+", default=[], help="pnp: reference가 빈 카메라용 영상 (파일 이름·경로·번호)")
    ap.add_argument("--pcd", nargs="+", default=[], help="pnp: reference가 빈 카메라용 포인트 클라우드 데이터")
    ap.add_argument("--match", choices=["order", "number"], default="order", help="pnp: --img·--pcd 짝짓기 (순서 / 번호)")
    args = ap.parse_args()

    cfg = load_cfg()
    run = {"pnp": step_pnp, "measure": step_measure, "kalman": step_kalman, "render": step_render}
    for s in STEPS:
        if s in args.steps:
            t0 = time.time()
            print(f"\n===== {s} =====", flush=True)
            run[s](cfg, args)
            print(f"({s} {time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
