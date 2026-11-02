"""Segments.ai 연결: API 키 찾기, 클라이언트, 재시도 업로드."""
import time

from segments import SegmentsClient


def resolve_api_key(cfg, override=None):
    """--api-key > upload_config.yaml의 api_key 순으로 API 키를 찾는다."""
    key = override or cfg.get("api_key")
    if not key:
        raise SystemExit("API 키가 필요합니다: configs/upload_config.yaml의 api_key에 적거나 --api-key로 넘기세요.")
    return key


def get_client(cfg, override=None):
    return SegmentsClient(resolve_api_key(cfg, override))


def upload_with_retry(client, file_path, filename, max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                return client.upload_asset(f, filename=filename)
        except Exception:
            if attempt == max_retries - 1:
                raise
            print(f"  업로드 실패 ({attempt + 1}/{max_retries}), {delay}초 후 재시도")
            time.sleep(delay)
