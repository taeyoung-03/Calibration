"""대응점 파일 (configs/calibration_points.yaml) 읽기·기록. camera_point.py·lidar_point.py의 --save, calibration.py가 쓴다."""
import os

from .config import CALIB_POINTS_PATH
from .yaml_io import load_yaml, save_yaml

HEADER = """\
# 카메라별 기준 프레임과 2D–3D 대응점 (calibration.py가 읽음).
#   reference.<카메라>: 대응점을 고른 영상(img)과 포인트 클라우드 데이터(pcd) 파일 이름.
#     번호만 써도 됨 ("2401" → 002401.jpg / 002401.pcd). 이름만 쓰면 data_root(CAMERA/<카메라>/, LiDAR/)에서 찾고, 경로를 쓰면 그대로 씀.
#     비워 두면 calibration.py --img ... --pcd ... 인자로 받고, 인자도 없으면 그 카메라는 기존 외부 파라미터를 그대로 둠.
#     기준 프레임은 시간 방향 처리(kalman)에서 보정 0으로 고정됨.
#   <카메라>.lidar_pts: 포인트 클라우드에서 고른 3D 좌표 [x, y, z] (lidar_point.py)
#   <카메라>.image_pts: 같은 순서로 영상에서 고른 픽셀 좌표 [u, v] (camera_point.py). 카메라당 6~10쌍 권장
"""


def load_points(path=CALIB_POINTS_PATH):
    return load_yaml(path) if os.path.exists(path) else {}


def save_points(cam, key, pts, ref_key, ref_value, path=CALIB_POINTS_PATH):
    """cam의 key(image_pts / lidar_pts)를 pts로 바꾸고, reference.cam.ref_key(img / pcd)를 ref_value로 기록. 나머지 항목은 그대로."""
    data = load_points(path)
    refs = data.setdefault("reference", {})
    refs.setdefault(cam, {"img": "", "pcd": ""})[ref_key] = str(ref_value)
    entry = data.setdefault(cam, {})
    entry[key] = [[round(float(x), 6) if key == "lidar_pts" else int(round(x)) for x in p] for p in pts]
    other = "lidar_pts" if key == "image_pts" else "image_pts"
    entry.setdefault(other, [])
    data = {"reference": data.pop("reference"), **data}
    save_yaml(path, data, header=HEADER)
    return len(entry[key]), len(entry[other])
