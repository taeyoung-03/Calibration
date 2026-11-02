"""YAML 읽기·쓰기. 숫자만 든 리스트(대응점 좌표 등)는 한 줄([x, y, z])로 쓴다."""
import os

import yaml


class _Dumper(yaml.SafeDumper):
    pass


def _represent_list(dumper, data):
    flow = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data, header=""):
    """data를 path에 저장. header가 있으면 파일 맨 위에 그대로 붙인다 (주석).
    임시 파일에 다 쓴 뒤 바꿔 끼우므로, 쓰는 도중에 멈춰도 원래 파일이 깨지지 않음."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(data, f, Dumper=_Dumper, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def update_yaml_entry(path, key, value):
    """YAML 파일에서 key 항목만 value로 바꾸고 나머지는 그대로 저장."""
    data = load_yaml(path) if os.path.exists(path) else {}
    data[key] = value
    save_yaml(path, data)
