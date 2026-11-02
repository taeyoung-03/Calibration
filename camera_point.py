#!/usr/bin/env python3
"""영상에서 정합용 2D 대응점(픽셀 좌표)을 클릭으로 고른다.

입력: 카메라 영상 (--cam·--frame이면 데이터 폴더의 CAMERA/<카메라>/<프레임>.jpg, --img면 그 파일)
출력: 고른 점의 픽셀 좌표 [u, v] 목록 (터미널). --save면 configs/calibration_points.yaml의
      <카메라>.image_pts와 reference.<카메라>.img에 기록 (나머지 항목은 그대로)

조작: 왼쪽 클릭 = 점 추가, 오른쪽 클릭 또는 z = 마지막 점 취소, q 또는 ESC = 끝내기
      lidar_point.py에서 고를 3D 점과 같은 순서로 고를 것

사용법:
    python3 camera_point.py --cam camera1 --frame 2401
    python3 camera_point.py --cam camera1 --frame 2401 --save
    python3 camera_point.py --img path/to/000001.jpg
"""
import argparse
import os

import cv2

from common.config import img_path, load_cfg
from common.points import save_points

WIN = "camera_point (left: add, right/z: undo, q/ESC: quit)"


def main():
    ap = argparse.ArgumentParser(description="영상 2D 대응점 클릭")
    ap.add_argument("--cam", default=None, help="카메라 이름 (예: camera1). --save에 필요")
    ap.add_argument("--frame", type=int, default=None, help="프레임 번호 (데이터 폴더에서 영상을 찾음)")
    ap.add_argument("--img", default=None, help="영상 파일 경로 (--cam·--frame 대신)")
    ap.add_argument("--save", action="store_true", help="calibration_points.yaml에 기록")
    args = ap.parse_args()

    if args.img:
        path = args.img
    elif args.cam and args.frame is not None:
        path = img_path(load_cfg(), args.cam, args.frame)
    else:
        raise SystemExit("--img 또는 --cam과 --frame을 지정하세요")
    if args.save and not args.cam:
        raise SystemExit("--save에는 --cam이 필요합니다")
    base = cv2.imread(path)
    if base is None:
        raise SystemExit(f"영상 읽기 실패: {path}")

    points = []

    def redraw():
        img = base.copy()
        for i, (u, v) in enumerate(points, 1):
            cv2.circle(img, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(img, f"P{i} ({u},{v})", (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WIN, img)

    def undo():
        if points:
            print(f"P{len(points)} 취소")
            points.pop()
            redraw()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"P{len(points)}: ({x}, {y})")
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            undo()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)                     # 창 크기를 줄여도 클릭 좌표는 원본 픽셀
    cv2.resizeWindow(WIN, min(base.shape[1], 1600), min(base.shape[0], 900))
    cv2.setMouseCallback(WIN, click)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("z"):
            undo()
    cv2.destroyAllWindows()

    print(f"\n{os.path.basename(path)}: {len(points)}개")
    print("image_pts:")
    for u, v in points:
        print(f"  - [{u}, {v}]")
    if args.save and points:
        ref = str(args.frame) if args.frame is not None else args.img     # 번호 또는 파일 경로 (calibration.py가 둘 다 읽음)
        n_img, n_lidar = save_points(args.cam, "image_pts", points, "img", ref)
        print(f"→ configs/calibration_points.yaml {args.cam}.image_pts {n_img}개 (lidar_pts {n_lidar}개), reference.img = {ref}")
        if n_lidar and n_lidar != n_img:
            print("  ⚠️ image_pts와 lidar_pts 개수가 다름 — 같은 순서·같은 개수로 맞춰야 calibration.py가 동작")


if __name__ == "__main__":
    main()
