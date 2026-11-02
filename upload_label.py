#!/usr/bin/env python3
"""TXT 검출 결과(3D 박스)를 Segments.ai 샘플의 라벨로 업로드한다.

입력: TXT 파일 (한 줄 = frame_name class x y z dx dy dz yaw, 공백 구분. 예: 000001 boat 10 10 10 5 10 3 2.1)
      configs/upload_config.yaml (dataset, API 키, label.sensor_name, label.labelset)
      데이터(포인트 클라우드 + 영상)는 upload_to_segments.py로 이미 올라가 있어야 함.
      프레임 순서·센서 목록·카테고리·샘플 uuid는 Segments.ai API에서 조회 (샘플은 --sample-name, 없으면 첫 번째 샘플)
출력: Segments.ai 라벨 (labelset이 없으면 새로 만듦). --output이면 라벨 attributes json도 저장
      --dry-run이면 업로드 없이 attributes만 만듦 (조회를 위해 API 키는 필요)

사용법:
    python3 upload_label.py --txt detection_test.txt --dry-run --output out/label_check.json
    python3 upload_label.py --txt detection_test.txt
"""
import argparse
import json
import os
from collections import defaultdict

from common.config import UPLOAD_CONFIG_PATH
from common.yaml_io import load_yaml


def parse_txt(path):
    """frame_name → [(class, x, y, z, dx, dy, dz, yaw), ...]"""
    detections = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 9:
                raise ValueError(f"{path}:{line_no} 열 수가 9개가 아님 (frame_name class x y z dx dy dz yaw): {line}")
            detections[parts[0]].append((parts[1], *(float(v) for v in parts[2:])))
    return detections


def fetch_reference(client, dataset, sensor_name="Lidar", sample_name=None):
    """Segments.ai에서 프레임 순서, 센서 목록, 카테고리, 샘플 uuid 조회.
    get_samples()는 attributes를 비워 주므로 uuid만 찾고, 센서·프레임은 get_sample(uuid)로 따로 조회."""
    samples = client.get_samples(dataset)
    if not samples:
        raise SystemExit(f"데이터셋 '{dataset}'에 샘플이 없습니다.")
    if sample_name:
        summary = next((s for s in samples if s.name == sample_name), None)
        if summary is None:
            raise SystemExit(f"이름이 '{sample_name}'인 샘플이 없습니다. 있는 샘플: {[s.name for s in samples][:10]}")
    else:
        summary = samples[0]
    sample = client.get_sample(summary.uuid)
    sensors = sample.attributes.sensors
    lidar = next((s for s in sensors if s.name == sensor_name), None)
    if lidar is None:
        raise SystemExit(f"센서 '{sensor_name}'가 없습니다. 있는 센서: {[s.name for s in sensors]}")
    frame_names = [f.name for f in lidar.attributes.frames]
    cat_map = {c.name: c.id for c in client.get_dataset(dataset).task_attributes.categories}
    return frame_names, [(s.name, s.task_type) for s in sensors], cat_map, sample.uuid, sample.name


def build_label_attributes(detections, frame_names, sensor_order, cat_map, sensor_name="Lidar"):
    """검출 결과 → Segments.ai 라벨 attributes (LiDAR 센서에 cuboid, 나머지 센서는 빈 라벨)."""
    unmatched = set(detections) - set(frame_names)
    if unmatched:
        print(f"경고: 샘플에 없는 frame_name {len(unmatched)}개: {sorted(unmatched)[:10]}")
    next_id, n_with_box, lidar_frames = 1, 0, []
    for idx, fid in enumerate(frame_names):
        annotations = []
        for cls, x, y, z, dx, dy, dz, yaw in detections.get(fid, []):
            if cls not in cat_map:
                raise SystemExit(f"카테고리 '{cls}'가 데이터셋에 없습니다. 있는 카테고리: {list(cat_map)}")
            annotations.append({"track_id": next_id, "id": next_id, "category_id": cat_map[cls], "type": "cuboid",
                                "position": {"x": x, "y": y, "z": z}, "dimensions": {"x": dx, "y": dy, "z": dz},
                                "yaw": yaw, "is_keyframe": True, "index": idx})
            next_id += 1
        n_with_box += bool(annotations)
        lidar_frames.append({"annotations": annotations})
    sensors = [{"name": name, "task_type": task_type,
                "attributes": {"format_version": "0.2",
                               "frames": lidar_frames if name == sensor_name else [{"annotations": []} for _ in frame_names]}}
               for name, task_type in sensor_order]
    print(f"프레임 {len(frame_names)}개 중 박스가 있는 프레임 {n_with_box}개, 박스 {next_id - 1}개")
    return {"sensors": sensors}


def main():
    ap = argparse.ArgumentParser(description="TXT 검출 결과 → Segments.ai 라벨 업로드")
    ap.add_argument("--txt", required=True, help="검출 결과 TXT")
    ap.add_argument("--config", default=UPLOAD_CONFIG_PATH, help="upload_config.yaml 경로")
    ap.add_argument("--dataset", default=None, help="설정의 dataset 대신 (\"owner/dataset_name\")")
    ap.add_argument("--api-key", default=None, help="upload_config.yaml의 api_key 대신 직접 API 키")
    ap.add_argument("--sensor-name", default=None, help="기본: 설정의 label.sensor_name")
    ap.add_argument("--sample-name", default=None, help="라벨을 붙일 샘플 이름 (기본: 첫 번째 샘플)")
    ap.add_argument("--labelset", default=None, help="기본: 설정의 label.labelset")
    ap.add_argument("--output", default=None, help="만든 attributes json 저장 경로 (선택)")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 attributes만 만듦")
    args = ap.parse_args()

    U = load_yaml(args.config)
    dataset = args.dataset or U["dataset"]
    sensor_name = args.sensor_name or U["label"]["sensor_name"]
    labelset = args.labelset or U["label"]["labelset"]

    from common.segments_client import get_client
    client = get_client(U, override=args.api_key)
    print(f"'{dataset}'에서 샘플·센서·카테고리 조회 중")
    frame_names, sensor_order, cat_map, sample_uuid, sample_name = fetch_reference(client, dataset, sensor_name, args.sample_name)
    print(f"대상 샘플: {sample_name} ({sample_uuid}), 프레임 {len(frame_names)}개, 센서 {[n for n, _ in sensor_order]}")
    attributes = build_label_attributes(parse_txt(args.txt), frame_names, sensor_order, cat_map, sensor_name)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(attributes, f, ensure_ascii=False, indent=2)
        print(f"attributes json → {args.output}")
    if args.dry_run:
        print("[dry-run] 업로드하지 않음")
        return
    if labelset not in {ls.name for ls in client.get_labelsets(dataset)}:
        print(f"labelset '{labelset}'이 없어 새로 만듦")
        client.add_labelset(dataset, labelset)
    client.add_label(sample_uuid, labelset, attributes)
    print(f"업로드 완료: 샘플 {sample_name} ({sample_uuid}), labelset {labelset}")


if __name__ == "__main__":
    main()
