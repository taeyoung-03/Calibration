#!/usr/bin/env python3
"""카메라 영상과 포인트 클라우드 데이터를 프레임별 외부 파라미터와 함께 Segments.ai에 업로드한다 (다중 센서 시퀀스 샘플 1개).

입력: 데이터 폴더(calibration_config.yaml의 data_root)의 LiDAR/<프레임>.pcd, CAMERA/<카메라>/<프레임>.jpg
      configs/camera_intrinsic.yaml, configs/camera_lidar_extrinsic.yaml, configs/upload_config.yaml
      외부 파라미터: correction: kalman이면 camera_lidar_extrinsic.yaml의 frames(프레임별 값, calibration.py kalman 결과),
      그 프레임·카메라 값이 없거나 correction: none이면 기준 외부 파라미터. 업로드 직전에 카메라→LiDAR로 역변환 + 쿼터니언
출력: Segments.ai 데이터셋의 샘플 (upload_config.yaml의 dataset, sample_name)
      --dry-run이면 업로드 없이 out/upload_preview.json (프레임·카메라별 Segments.ai 형식 외부 파라미터)

API 키: upload_config.yaml의 api_key (--api-key로 주면 그 값이 우선)

사용법:
    python3 upload_to_segments.py --dry-run     # 업로드 없이 확인
    python3 upload_to_segments.py               # 업로드
"""
import argparse
import json
import os

from common.camera import distortion_dict, intrinsic_matrix_list, load_intrinsics
from common.config import OUT, UPLOAD_CONFIG_PATH, load_cfg
from common.extrinsics import load_extrinsics, segments_extrinsics
from common.yaml_io import load_yaml


def image_entry(cam, url, row, col, I, ext):
    return {"name": cam, "url": url, "row": row, "col": col,
            "intrinsics": {"intrinsic_matrix": intrinsic_matrix_list(I)},
            "distortion": distortion_dict(I),
            "extrinsics": segments_extrinsics(ext),
            "camera_convention": "OpenCV"}


def main():
    ap = argparse.ArgumentParser(description="Segments.ai 업로드 (프레임별 외부 파라미터 반영)")
    ap.add_argument("--config", default=UPLOAD_CONFIG_PATH, help="upload_config.yaml 경로")
    ap.add_argument("--dataset", default=None, help="설정의 dataset 대신 (\"owner/dataset_name\")")
    ap.add_argument("--api-key", default=None, help="upload_config.yaml의 api_key 대신 직접 API 키")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 프레임별 외부 파라미터만 out/upload_preview.json으로")
    args = ap.parse_args()

    U, cfg = load_yaml(args.config), load_cfg()
    dataset, cams, layout = args.dataset or U["dataset"], U["cameras"], U["cam_layout"]
    intr = load_intrinsics()
    base, _, frame_ext = load_extrinsics()
    if U.get("correction", "kalman") != "kalman":
        frame_ext = {}
    fi = U["frame_index"]
    frames = [f"{n:06d}" for n in range(fi["start"], fi["end"] + 1, fi["step"]) if n not in set(fi.get("exclude") or [])]
    missing = [c for c in cams if c not in base]
    if missing:
        raise SystemExit(f"기준 외부 파라미터 없음: {missing} → calibration.py를 먼저 실행하세요")

    plan, n_frame = {}, 0                                       # (프레임, 카메라) → 외부 파라미터 (LiDAR→카메라)
    for name in frames:
        for cam in cams:
            per_frame = cam in frame_ext.get(name, {})
            plan[(name, cam)] = frame_ext[name][cam] if per_frame else base[cam]
            n_frame += per_frame
    print(f"{len(frames)}프레임 × {len(cams)}카메라: 프레임별 외부 파라미터 {n_frame}개, 기준 외부 파라미터 {len(plan) - n_frame}개")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "upload_preview.json"), "w") as f:
        json.dump({f"{n}/{c}": segments_extrinsics(e) for (n, c), e in plan.items()}, f, indent=1)
    if args.dry_run:
        print(f"[dry-run] 업로드하지 않음. Segments.ai 형식 외부 파라미터 → {os.path.join(OUT, 'upload_preview.json')}")
        return

    from common.segments_client import get_client, upload_with_retry
    client = get_client(U, override=args.api_key)
    lidar_frames = []
    for i, name in enumerate(frames, 1):
        pcd = os.path.join(cfg["data_root"], "LiDAR", f"{name}.pcd")
        if not os.path.exists(pcd):
            print(f"{name}: 포인트 클라우드 데이터 없음 — 건너뜀")
            continue
        print(f"[{i}/{len(frames)}] {name} 업로드 중", flush=True)
        asset = upload_with_retry(client, pcd, f"{name}.pcd")
        images = []
        for cam in cams:
            img = os.path.join(cfg["data_root"], "CAMERA", cam, f"{name}.jpg")
            if not os.path.exists(img):
                print(f"  {cam} 영상 없음")
                continue
            a = upload_with_retry(client, img, f"{cam}_{name}.jpg")
            images.append(image_entry(cam, a.url, *layout[cam], intr[cam], plan[(name, cam)]))
        lidar_frames.append({"name": name, "pcd": {"url": asset.url, "type": "pcd"}, "images": images})

    sensors = [{"name": "Lidar", "task_type": "pointcloud-cuboid-sequence", "attributes": {"frames": lidar_frames}}]
    for cam in cams:
        sensors.append({"name": cam, "task_type": "image-vector-sequence", "attributes": {"frames": [
            {"image": {"url": im["url"]}} for f in lidar_frames for im in f["images"] if im["name"] == cam]}})
    sample = client.add_sample(dataset, U["sample_name"], {"sensors": sensors})
    print(f"\n완료. Sample UUID: {sample.uuid}\n  https://segments.ai/{dataset}/samples/{sample.uuid}")


if __name__ == "__main__":
    main()
