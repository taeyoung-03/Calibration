"""설정·경로: 저장소 폴더 기준 경로, 정합 설정(calibration_config.yaml) 읽기, 처리 프레임 목록, 데이터 파일 경로."""
import os

from .yaml_io import load_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")
OUT = os.path.join(ROOT, "out")
CALIB_CONFIG_PATH = os.path.join(CONFIG_DIR, "calibration_config.yaml")
INTRINSICS_PATH = os.path.join(CONFIG_DIR, "camera_intrinsic.yaml")
EXTRINSICS_PATH = os.path.join(CONFIG_DIR, "camera_lidar_extrinsic.yaml")  # 카메라 키 = 기준, reference = 기준 프레임, frames = 프레임별 값
CALIB_POINTS_PATH = os.path.join(CONFIG_DIR, "calibration_points.yaml")
UPLOAD_CONFIG_PATH = os.path.join(CONFIG_DIR, "upload_config.yaml")


def resolve(path):
    """'~'·저장소 기준 상대 경로 → 절대 경로."""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_cfg(path=CALIB_CONFIG_PATH):
    cfg = load_yaml(path)
    cfg["data_root"] = resolve(cfg["data_root"])
    os.makedirs(OUT, exist_ok=True)
    return cfg


def frame_list(cfg, reference=()):
    """설정의 프레임 간격 목록 + 카메라별 기준 프레임 (간격에 안 맞아도 포함)."""
    f = cfg["frames"]
    refs = {n for n in reference if f["start"] <= n <= f["end"]}
    return sorted(set(range(f["start"], f["end"] + 1, f["step"])) | refs)


def frame_number(path):
    """파일 이름의 숫자 → 프레임 번호 (예: .../000003.jpg → 3)."""
    digits = "".join(ch for ch in os.path.splitext(os.path.basename(str(path)))[0] if ch.isdigit())
    return int(digits) if digits else None


def data_file(cfg, path, folder, ext):
    """파일 이름·번호 → 경로. "2401"처럼 번호만 쓰면 6자리 + 확장자를 붙이고, 이름만 쓰면 데이터 폴더의 folder에서 찾는다."""
    path = str(path)
    if path.isdigit():
        path = f"{int(path):06d}{ext}"
    elif not os.path.splitext(path)[1]:
        path += ext
    return os.path.expanduser(path) if os.path.dirname(path) else os.path.join(cfg["data_root"], folder, path)


def img_path(cfg, cam, n):
    return os.path.join(cfg["data_root"], "CAMERA", cam, f"{n:06d}.jpg")


def pcd_path(cfg, n):
    return os.path.join(cfg["data_root"], "LiDAR", f"{n:06d}.pcd")


def depth_path(cfg, cam, n):
    """depth_map.py가 만든 깊이 맵 (16bit png). 폴더는 설정 depth.dir (out/ 아래)."""
    return os.path.join(OUT, cfg["depth"]["dir"], cam, f"{n:06d}.png")
