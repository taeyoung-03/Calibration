#!/usr/bin/env python3
"""포인트 클라우드 데이터에서 정합용 3D 대응점(LiDAR 좌표)을 고른다 (Open3D 편집 창).

입력: 포인트 클라우드 데이터 (--frame이면 데이터 폴더의 LiDAR/<프레임>.pcd, --pcd면 그 파일)
출력: 고른 점의 LiDAR 좌표 [x, y, z] 목록 (터미널). --save면 configs/calibration_points.yaml의
      <카메라>.lidar_pts와 reference.<카메라>.pcd에 기록 (나머지 항목은 그대로)

조작 (Open3D): Shift + 왼쪽 클릭 = 점 선택, Shift + 오른쪽 클릭 = 마지막 선택 취소, Q = 끝내기
      마우스 휠·드래그로 시점 이동. 포인트 색 = 반사 강도. camera_point.py에서 고른 2D 점과 같은 순서로 고를 것

사용법:
    python3 lidar_point.py --frame 2401
    python3 lidar_point.py --frame 2401 --cam camera1 --save
    python3 lidar_point.py --pcd path/to/000001.pcd
"""
import argparse
import os

import cv2
import numpy as np

from common.config import load_cfg, pcd_path
from common.pcd_io import read_pcd
from common.points import save_points


def intensity_colors(val):
    """반사 강도 → RGB [0, 1] (log 후 1~99%를 컬러맵으로)."""
    v = np.log1p(np.maximum(val, 0))
    lo, hi = np.percentile(v, [1, 99])
    g = (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1] / 255.0


def main():
    ap = argparse.ArgumentParser(description="포인트 클라우드 데이터 3D 대응점 선택")
    ap.add_argument("--frame", type=int, default=None, help="프레임 번호 (데이터 폴더에서 찾음)")
    ap.add_argument("--pcd", default=None, help="포인트 클라우드 데이터 파일 경로 (--frame 대신)")
    ap.add_argument("--cam", default=None, help="카메라 이름. --save에 필요")
    ap.add_argument("--save", action="store_true", help="calibration_points.yaml에 기록")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.pcd:
        path = args.pcd
    elif args.frame is not None:
        path = pcd_path(cfg, args.frame)
    else:
        raise SystemExit("--pcd 또는 --frame을 지정하세요")
    if args.save and not args.cam:
        raise SystemExit("--save에는 --cam이 필요합니다")
    xyz, val = read_pcd(path, cfg["lidar"]["field"], cfg["lidar"]["min_range"])   # 반사 없는 칸·선체 포인트 제외
    print(f"{os.path.basename(path)}: 포인트 {len(xyz)}개")

    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(intensity_colors(val))
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="lidar_point (Shift+left: pick, Shift+right: undo, Q: quit)", width=1600, height=900)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    pts = xyz[picked] if len(picked) else np.zeros((0, 3))
    print(f"\n{len(pts)}개")
    print("lidar_pts:")
    for x, y, z in pts:
        print(f"  - [{x:.4f}, {y:.4f}, {z:.4f}]")
    if args.save and len(pts):
        ref = str(args.frame) if args.frame is not None else args.pcd     # 번호 또는 파일 경로 (calibration.py가 둘 다 읽음)
        n_lidar, n_img = save_points(args.cam, "lidar_pts", pts.tolist(), "pcd", ref)
        print(f"→ configs/calibration_points.yaml {args.cam}.lidar_pts {n_lidar}개 (image_pts {n_img}개), reference.pcd = {ref}")
        if n_img and n_img != n_lidar:
            print("  ⚠️ lidar_pts와 image_pts 개수가 다름 — 같은 순서·같은 개수로 맞춰야 calibration.py가 동작")


if __name__ == "__main__":
    main()
