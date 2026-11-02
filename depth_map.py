#!/usr/bin/env python3
"""카메라 영상 → 단안 깊이 맵 (Depth Anything V2 Small, 상대 역깊이 = 가까울수록 큼). calibration.py measure의 Δu 측정에 쓴다.

입력: 데이터 폴더의 CAMERA/<카메라>/<프레임>.jpg (설정 frames의 프레임 + 카메라별 기준 프레임), 모델 가중치 checkpoints/depth_anything_v2_vits.pth
출력: out/<depth.dir>/<카메라>/<프레임>.png  (16bit, 영상마다 최소~최대를 0~65535로. 크기는 원본 × depth.save_scale)
이미 있는 파일은 건너뛴다 (다시 만들려면 그 폴더를 지울 것). GPU(CUDA) 필요.

사용법:
    python3 depth_map.py
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common.config import OUT, ROOT, depth_path, frame_list, img_path, load_cfg, resolve
from common.extrinsics import load_extrinsics

sys.path.insert(0, os.path.join(ROOT, "third_party", "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize  # noqa: E402
from depth_anything_v2.dinov2_layers.attention import Attention  # noqa: E402

VITS = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}   # Small만 사용 (Base·Large·Giant는 CC-BY-NC-4.0)


def _sdpa_forward(self, x):
    """Attention.forward와 같은 계산 (추론, dropout 없음)을 PyTorch 내장 scaled_dot_product_attention으로.
    입력이 클 때(input_size 1080) 어텐션 행렬을 통째로 만들지 않아 빠름."""
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
    return self.proj(x)


Attention.forward = _sdpa_forward


def infer(model, img, input_size, out_hw):
    """model.infer_image()와 같은 전처리·추론 → out_hw 크기 float32 역깊이.
    JetPack의 torch가 NumPy 2와 호환이 안 되는 경우가 있어(from_numpy·.numpy() 불가) numpy ↔ torch를 바이트로 넘긴다."""
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    for t in (Resize(input_size, input_size, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                     resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
              NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()):
        x = t({"image": x})["image"]
    x = np.ascontiguousarray(x, np.float32)
    xt = torch.frombuffer(bytearray(x.tobytes()), dtype=torch.float32).reshape(1, *x.shape).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        d = model(xt)
    d = F.interpolate(d[:, None].float(), out_hw, mode="bilinear", align_corners=True)[0, 0]
    return np.array(d.cpu().flatten().tolist(), np.float32).reshape(out_hw)


def main():
    argparse.ArgumentParser(description="카메라 영상 → 깊이 맵 (Depth Anything V2 Small). 설정: configs/calibration_config.yaml").parse_args()
    cfg = load_cfg()
    D = cfg["depth"]
    ckpt = resolve(D["checkpoint"])
    if not os.path.exists(ckpt):
        raise SystemExit(f"모델 가중치 없음: {ckpt} (README의 설치 절차 참고)")
    model = DepthAnythingV2(**VITS)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to("cuda").eval()

    _, ref, _ = load_extrinsics()
    jobs = [(cam, n) for n in frame_list(cfg, ref.values()) for cam in cfg["cameras"]
            if not os.path.exists(depth_path(cfg, cam, n))]
    print(f"깊이 맵 {len(jobs)}장 (Depth Anything V2 Small, 입력 {D['input_size']})", flush=True)
    t0, last = time.time(), time.time()
    for i, (cam, n) in enumerate(jobs, 1):
        img = cv2.imread(img_path(cfg, cam, n))
        if img is None:                                         # 없는 영상, 손상된 JPG
            continue
        h, w = img.shape[:2]
        d = infer(model, img, D["input_size"], (int(h * D["save_scale"]), int(w * D["save_scale"])))
        d = (d - d.min()) / max(np.ptp(d), 1e-6)
        os.makedirs(os.path.dirname(depth_path(cfg, cam, n)), exist_ok=True)
        cv2.imwrite(depth_path(cfg, cam, n), (d * 65535).astype(np.uint16))
        if time.time() - last > 60 or i == len(jobs):
            last = time.time()
            print(f"{i}/{len(jobs)} ({i / (last - t0):.1f}장/s)", flush=True)
    print("저장 →", os.path.join(OUT, D["dir"]))


if __name__ == "__main__":
    main()
