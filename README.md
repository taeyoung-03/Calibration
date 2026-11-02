# LiDAR–Camera Frame-wise Extrinsic Calibration

본 저장소는 LiDAR 포인트 클라우드와 카메라 영상의 외부 정합(Extrinsic Calibration)을 수행하고, 프레임마다 생기는 어긋남을 보정하여 프레임·카메라별 외부 파라미터를 만드는 코드입니다.  
정합 결과는 Segments.ai 라벨링 도구에 함께 업로드하여, 영상으로 객체를 확인하면서 포인트 클라우드에 3차원 라벨을 붙이는 데 사용합니다.

기존 `segments-ai` 저장소의 기능(대응점 선택, solvePnP 정합, Segments.ai 데이터·라벨 업로드)을 모두 포함하며, 여기에 프레임별 보정을 더했습니다.

정합은 두 단계로 이루어집니다.

1. **기준 외부 파라미터**: 사람이 영상과 포인트 클라우드에서 고른 2D–3D 대응점으로 `solvePnPRansac`을 풀어 카메라별 외부 파라미터 R, t를 구합니다.
2. **프레임별 보정**: 프레임마다 투영한 포인트 클라우드 전체를 (Δu, Δv) px만큼 옮겨 보며 영상과 가장 잘 맞는 위치를 측정합니다.
   - Δv(세로): LiDAR 반사 강도와 영상 밝기의 NID로 먼저 구합니다.
   - Δu(가로): 그 Δv의 가로선에서, 단안 깊이 맵(Depth Anything V2 Small)이 LiDAR 거리와 분명히 모순되는 경우(하늘에 떨어진 포인트, 경계 모순 쌍, 거리별 정상 범위를 벗어난 포인트)가 가장 적은 위치를 찾습니다.
   - 측정값은 칼만 필터 + RTS smoother로 시간 방향으로 정리하고, 작은 회전 ΔR = Rx(−atan(Δv/fy))·Ry(atan(Δu/fx))로 바꿔 R′ = ΔR·R, t′ = ΔR·t로 적용합니다.

## Repository Structure

```text
Calibration/
├── common/
│   ├── camera.py
│   ├── config.py
│   ├── extrinsics.py
│   ├── pcd_io.py
│   ├── points.py
│   ├── projection.py
│   ├── segments_client.py
│   └── yaml_io.py
├── configs/
│   ├── calibration_config.yaml
│   ├── calibration_points.yaml
│   ├── camera_intrinsic.yaml
│   ├── camera_lidar_extrinsic.yaml
│   └── upload_config.yaml
├── third_party/
│   └── Depth-Anything-V2/
├── checkpoints/               # 깊이 모델 가중치 (depth_anything_v2_vits.pth)
├── label_sample/              # 데이터셋 (git 제외)
│   ├── CAMERA/
│   │   ├── camera1/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   ├── camera2/
│   │   ├── camera3/
│   │   └── camera6/
│   └── LiDAR/
│       ├── 000001.pcd
│       └── ...
├── out/                       # 실행 결과 (git 제외)
├── calibration.py
├── camera_point.py
├── depth_map.py
├── lidar_point.py
├── upload_label.py
├── upload_to_segments.py
├── detection_test.txt
└── requirements.txt
```

## Installation

개발·검증 환경은 NVIDIA Jetson AGX Orin 64GB, JetPack 6.2.3 (Ubuntu 22.04), Python 3.10.12입니다.

```bash
sudo apt update
sudo apt install -y python3-pip git
pip3 install -r requirements.txt
```

깊이 맵 생성(`depth_map.py`)에는 CUDA를 쓰는 PyTorch와 torchvision이 추가로 필요합니다. 플랫폼마다 설치 방법이 달라 `requirements.txt`에는 넣지 않았습니다.

- Jetson: JetPack 버전에 맞는 NVIDIA 제공 PyTorch 휠을 설치합니다 (확인한 버전: torch 2.8.0, torchvision 0.23.0).
- x86 PC: PyTorch 공식 안내에 따라 CUDA 버전에 맞는 torch, torchvision을 설치합니다.

깊이 모델은 Depth Anything V2 **Small** 가중치만 사용합니다 (Apache-2.0). Base·Large·Giant는 비상업 라이선스(CC-BY-NC-4.0)이므로 사용하지 않습니다. 가중치 `checkpoints/depth_anything_v2_vits.pth`는 저장소에 들어 있어 따로 받을 필요가 없습니다. 파일이 없거나 손상됐으면 아래 명령으로 다시 받습니다.

```bash
wget -O checkpoints/depth_anything_v2_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
```

데이터는 `label_sample/` 아래에 두거나(`CAMERA/<카메라>/<번호 6자리>.jpg`, `LiDAR/<번호 6자리>.pcd`), `configs/calibration_config.yaml`의 `data_root`를 데이터 폴더 경로로 바꿉니다.

## Quick Start

```bash
# 1. 대응점 고르기 (카메라마다, 같은 프레임에서 같은 순서로 6~10쌍)
python3 camera_point.py --cam camera1 --frame 2401 --save
python3 lidar_point.py  --cam camera1 --frame 2401 --save

# 2. 깊이 맵 만들기 (처음 한 번, GPU 필요)
python3 depth_map.py

# 3. 정합 (기준 외부 파라미터 → 프레임별 측정 → 시간 방향 처리 → 확인 영상)
python3 calibration.py

# 4. Segments.ai 업로드 (configs/upload_config.yaml의 api_key에 키를 적고, 먼저 --dry-run으로 확인)
python3 upload_to_segments.py --dry-run
python3 upload_to_segments.py

# 5. (선택) 검출 결과 TXT를 라벨로 업로드
python3 upload_label.py --txt detection_test.txt
```

대응점을 고친 뒤에는 `calibration.py`를 처음부터 다시 실행해야 합니다. 프레임별 측정이 기준 외부 파라미터로 투영한 포인트 클라우드를 기준으로 하기 때문입니다.

## Folder Description

### common

여러 코드에서 공통으로 쓰는 함수 모듈이 포함된 폴더입니다.

- `config.py`: 저장소 기준 경로, `calibration_config.yaml` 읽기, 처리할 프레임 목록, 데이터 파일 경로
- `yaml_io.py`: YAML 읽기·쓰기 (특정 항목만 갱신)
- `camera.py`: 내부 파라미터(K 행렬, 왜곡 계수)와 Segments.ai 업로드 형식
- `extrinsics.py`: 외부 파라미터 변환 (yaw/pitch/roll·tx/ty/tz ↔ R, t), 프레임별 보정 회전, Segments.ai 형식(카메라→LiDAR 역변환 + 쿼터니언)
- `pcd_io.py`: 포인트 클라우드 데이터(binary PCD) 읽기. 반사가 없는 포인트(NaN)와 선체 포인트(가까운 거리)를 뺍니다.
- `projection.py`: LiDAR 포인트의 영상 투영, 보정 회전 후 다시 투영, 깊이 색 점 찍기
- `points.py`: 대응점 파일(`calibration_points.yaml`) 읽기·기록
- `segments_client.py`: Segments.ai API 키 찾기, 클라이언트, 재시도 업로드

### configs

정합과 업로드에 필요한 설정 파일과 정합 결과가 저장되는 폴더입니다.

- `calibration_config.yaml`  
  정합 설정 파일입니다. 데이터 폴더(`data_root`), 카메라 목록, 처리할 프레임 범위·간격, 병렬 프로세스 수, 깊이 모델 설정, 측정(`measure`)과 시간 방향 처리(`kalman`)의 기준값, 확인 영상(`render`) 설정이 들어 있습니다. 항목마다 의미를 주석으로 적었습니다.

- `calibration_points.yaml`  
  카메라별 2D–3D 대응점과 기준 프레임을 기록하는 파일입니다.  
  `camera_point.py --save`가 `image_pts`(영상 픽셀 좌표)를, `lidar_point.py --save`가 `lidar_pts`(LiDAR 좌표)를 기록하고, `calibration.py`가 읽습니다.  
  `reference`에는 대응점을 고른 영상과 포인트 클라우드 데이터 파일(번호만 써도 됨)을 적으며, 이 프레임은 시간 방향 처리에서 보정 0으로 고정됩니다.

- `camera_intrinsic.yaml`  
  카메라 내부 파라미터(해상도, fx, fy, cx, cy, 왜곡 계수 k1, k2, p1, p2, k3)를 저장하는 파일입니다.

- `camera_lidar_extrinsic.yaml`  
  Camera와 LiDAR 사이의 외부 정합 결과를 저장하는 파일입니다. `calibration.py`가 기록합니다.
  - `camera1` 등 카메라 키: 기준 외부 파라미터 (LiDAR→카메라, `yaw`·`pitch`·`roll` [rad, ZYX 오일러], `tx`·`ty`·`tz` [m])
  - `reference`: 카메라별 기준 프레임 번호
  - `frames`: 프레임·카메라별 외부 파라미터 (`"000001": {camera1: {yaw, pitch, roll, tx, ty, tz}, ...}`)

- `upload_config.yaml`  
  Segments.ai 업로드 설정 파일입니다. 데이터셋 이름, 샘플 이름, 업로드할 프레임 범위, 카메라 배치, 프레임별 외부 파라미터 사용 여부(`correction`), 라벨 업로드 설정이 들어 있습니다.  
  API 키는 이 파일의 `api_key`에 적습니다 (`--api-key`로 주면 그 값이 우선). 저장소에는 빈 값으로 올라가 있습니다.

### third_party

단안 깊이 추정 모델 Depth Anything V2의 모델 코드(Apache-2.0, `LICENSE` 포함)입니다. 추론에 필요한 `depth_anything_v2` 패키지만 들어 있습니다.

### checkpoints

깊이 모델 가중치(`depth_anything_v2_vits.pth`, 약 99MB, Apache-2.0)가 들어 있습니다. `depth_map.py`가 이 파일을 읽습니다.

### label_sample

데이터셋 폴더입니다 (`calibration_config.yaml`의 `data_root`). 저장소에는 올리지 않습니다.

- `CAMERA/`: 카메라 영상. `camera1`, `camera2`, `camera3`, `camera6` 폴더로 구분됩니다.
- `LiDAR/`: 포인트 클라우드 데이터 (binary PCD).

같은 번호의 영상 파일과 포인트 클라우드 데이터 파일은 같은 시점에 기록된 데이터입니다.

### out

실행 결과가 저장되는 폴더입니다. 저장소에는 올리지 않습니다.

| 경로 | 만드는 코드 | 내용 |
|---|---|---|
| `out/depth/<카메라>/<프레임>.png` | `depth_map.py` | 깊이 맵 (16bit, 가까울수록 큼) |
| `out/calib/<카메라>_<프레임>.jpg` | `calibration.py` pnp | 기준 외부 파라미터 투영 확인 영상 (초록 원 = 고른 영상 대응점, 빨간 십자 = 3D 대응점 재투영) |
| `out/measure.csv` | `calibration.py` measure | 프레임·카메라별 (Δu, Δv) 측정 |
| `out/kalman.csv` | `calibration.py` kalman | 프레임·카메라별 최종 보정값 |
| `out/proj/<카메라>/<프레임>.jpg` | `calibration.py` render | 프레임별 외부 파라미터로 투영한 확인 영상 |
| `out/upload_preview.json` | `upload_to_segments.py` | Segments.ai 형식 외부 파라미터 (업로드 전 확인용) |

## Python Scripts

### camera_point.py

카메라 영상에서 정합에 사용할 2D 대응점(픽셀 좌표)을 클릭으로 고르는 코드입니다.  
왼쪽 클릭으로 점을 추가하고, 오른쪽 클릭 또는 `z`로 마지막 점을 취소하며, `q` 또는 `ESC`로 끝냅니다.

- 입력: 카메라 영상 (`--cam`·`--frame`이면 데이터 폴더의 `CAMERA/<카메라>/<프레임>.jpg`, `--img`면 그 파일)
- 출력: 고른 점의 픽셀 좌표 목록 (터미널). `--save`면 `calibration_points.yaml`의 `<카메라>.image_pts`와 `reference.<카메라>.img`

```bash
python3 camera_point.py --cam camera1 --frame 2401 --save
python3 camera_point.py --img path/to/000001.jpg
```

### lidar_point.py

포인트 클라우드 데이터에서 정합에 사용할 3D 대응점(LiDAR 좌표)을 고르는 코드입니다 (Open3D 편집 창).  
Shift + 왼쪽 클릭으로 점을 고르고, Shift + 오른쪽 클릭으로 마지막 선택을 취소하며, `Q`로 끝냅니다. 포인트 색은 반사 강도입니다. `camera_point.py`에서 고른 점과 같은 순서로 고릅니다.

- 입력: 포인트 클라우드 데이터 (`--frame`이면 데이터 폴더의 `LiDAR/<프레임>.pcd`, `--pcd`면 그 파일)
- 출력: 고른 점의 LiDAR 좌표 목록 (터미널). `--save`면 `calibration_points.yaml`의 `<카메라>.lidar_pts`와 `reference.<카메라>.pcd`

```bash
python3 lidar_point.py --cam camera1 --frame 2401 --save
python3 lidar_point.py --pcd path/to/000001.pcd
```

### depth_map.py

카메라 영상에서 단안 깊이 맵(상대 역깊이)을 만드는 코드입니다. `calibration.py`의 Δu 측정에 사용합니다. GPU(CUDA)가 필요하며, 이미 있는 파일은 건너뜁니다.

- 입력: 데이터 폴더의 카메라 영상 (`calibration_config.yaml`의 `frames` 범위·간격 + 카메라별 기준 프레임), `checkpoints/depth_anything_v2_vits.pth`
- 출력: `out/depth/<카메라>/<프레임>.png`

```bash
python3 depth_map.py
```

Jetson AGX Orin에서 1920×1080 영상 2,332장(583프레임 × 카메라 4대)에 약 22분이 걸립니다.  
JetPack용 torch가 NumPy 1.x로 빌드되어 있으면 실행 시작 때 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0` 경고가 나옵니다. `depth_map.py`는 NumPy 배열과 torch 텐서를 바이트로 주고받아 이 경고와 관계없이 동작합니다.

### calibration.py

정합 전체를 실행하는 코드입니다. 네 단계를 순서대로 실행하며, `--steps`로 일부 단계만 실행할 수 있습니다.

| 단계 | 하는 일 | 입력 | 출력 |
|---|---|---|---|
| `pnp` | 대응점으로 `solvePnPRansac` (RANSAC 가설 EPnP → LM 정밀화, 재투영 오차 기준 8 px). `reference`가 있는 카메라만 다시 계산하고, 없는 카메라는 기존 값을 그대로 둠 | `calibration_points.yaml`, `camera_intrinsic.yaml` | `camera_lidar_extrinsic.yaml`의 카메라 키·`reference`, `out/calib/` |
| `measure` | 프레임·카메라마다 (Δu, Δv) 측정. Δv = 밝기 NID, Δu = 그 Δv의 가로선에서 깊이 맵 모순 세기. 탐색 범위 Δu ±50 px, Δv ±20 px | 데이터 폴더, `out/depth/`, 기준 외부 파라미터 | `out/measure.csv` |
| `kalman` | 고립·튐 측정 제거 → 칼만 필터 + RTS smoother → Δu 빈 구간은 다른 카메라 흐름 참고 → 프레임별 외부 파라미터 | `out/measure.csv`, 기준 외부 파라미터 | `out/kalman.csv`, `camera_lidar_extrinsic.yaml`의 `frames` |
| `render` | 프레임별 외부 파라미터로 포인트 클라우드를 투영한 확인 영상 (`render.step` 간격) | `out/kalman.csv`, 데이터 폴더 | `out/proj/` |

```bash
python3 calibration.py                                  # 전체
python3 calibration.py --steps kalman render            # 시간 방향 처리 설정만 바꿨을 때
python3 calibration.py --steps pnp --cam camera1 --img 2401 --pcd 2401   # reference가 빈 카메라의 기준 프레임을 인자로
```

Jetson AGX Orin(8 프로세스)에서 583프레임 × 카메라 4대 기준으로 measure 약 7분, pnp·kalman·render(59프레임)는 각각 10초 이내입니다.

`out/measure.csv` 열

| 열 | 의미 |
|---|---|
| `frame`, `camera` | 프레임 번호, 카메라 |
| `n_points` | 측정에 쓴 포인트 수 (500개 미만이면 측정하지 않고 du·dv가 빈칸) |
| `du`, `dv` | 측정한 보정값 (px) |
| `used_u`, `used_v` | 사용 판정 (1 = 칼만에 사용, 0 = 뚜렷하지 않거나 탐색 범위 끝) |
| `distinct_u`, `distinct_v` | 뚜렷함 (Δu: 다른 후보와의 점수 차 ÷ 점수 표준편차, Δv: 다른 후보와의 NID 차) |

`out/kalman.csv` 열

| 열 | 의미 |
|---|---|
| `kf_du`, `kf_dv` | 최종 보정값 (px). 이 값으로 프레임별 외부 파라미터를 만듦 |
| `sigma_u`, `sigma_v` | 칼만 표준편차 (px) |
| `spike_u`, `spike_v` | 튐으로 판정되어 뺀 측정이면 1 |
| `cross_u` | Δu 빈 구간에서 다른 카메라 흐름을 참고해 바꾼 프레임이면 1 |

### upload_to_segments.py

카메라 영상과 포인트 클라우드 데이터를 Segments.ai에 업로드하는 코드입니다 (다중 센서 시퀀스 샘플 1개).  
`correction: kalman`이면 프레임·카메라마다 `camera_lidar_extrinsic.yaml`의 `frames` 값을, 없으면 기준 외부 파라미터를 넣습니다. 업로드 직전에 Segments.ai 형식(카메라→LiDAR 역변환 + 쿼터니언, OpenCV 카메라 축)으로 바꿉니다.

- 입력: 데이터 폴더, `camera_intrinsic.yaml`, `camera_lidar_extrinsic.yaml`, `upload_config.yaml`, API 키
- 출력: Segments.ai 데이터셋의 샘플. `--dry-run`이면 업로드 없이 `out/upload_preview.json`

```bash
python3 upload_to_segments.py --dry-run
python3 upload_to_segments.py
```

### upload_label.py

TXT로 정리한 3D 박스 검출 결과를 Segments.ai 샘플의 라벨로 업로드하는 코드입니다. 데이터는 `upload_to_segments.py`로 먼저 올려 두어야 합니다. 프레임 순서, 센서 목록, 카테고리는 Segments.ai에서 조회하므로 `--dry-run`에도 API 키가 필요합니다.

- 입력: TXT 파일, `upload_config.yaml`(dataset, `label.sensor_name`, `label.labelset`), API 키
- 출력: Segments.ai 라벨 (labelset이 없으면 새로 만듦). `--output`이면 라벨 json도 저장

```bash
python3 upload_label.py --txt detection_test.txt --dry-run --output out/label_check.json
python3 upload_label.py --txt detection_test.txt --sample-name label_sample
```

## TXT File

### detection_test.txt

라벨 업로드 예제 파일입니다. 일부 라벨링 데이터로 학습한 모델이 예측한 3D 박스로, 라벨링 과정에서 보조 자료로 사용합니다.  
한 줄의 형식은 `frame_name class x y z dx dy dz yaw` (공백 구분)입니다.

## Notice

- 데이터셋(`label_sample/`), 실행 결과(`out/`), 캐시는 `.gitignore`로 저장소에서 제외합니다. 모델 가중치는 `checkpoints/depth_anything_v2_vits.pth`만 포함합니다.
- 가중치 파일(약 99MB)은 GitHub 파일당 제한(100MB)보다 작아 그대로 올라가지만, 50MB를 넘으므로 push 때 용량 경고가 나옵니다.
- 저장소의 `upload_config.yaml`은 `api_key`가 빈 값입니다. 각자 키를 적어 쓰고, 키를 적은 파일은 커밋하지 않습니다.
- `third_party/Depth-Anything-V2`는 Depth Anything V2 공식 저장소의 코드이며 Apache-2.0 라이선스를 따릅니다.
