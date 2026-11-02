"""포인트 클라우드 데이터(binary PCD) 읽기."""
import numpy as np

_NP = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4",
       ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def read_pcd(path, field="intensity", min_range=0.0):
    """binary PCD → (xyz (N, 3), 포인트 값 field (N,)). 반사 없는 칸(NaN)과 센서에서 min_range m 안의 포인트는 뺀다.
    field가 파일에 없으면 intensity를 쓴다."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD 헤더가 끝나지 않음: {path}")
            k, *v = line.decode().split() or [""]
            hdr[k] = v
            if k == "DATA":
                break
        if hdr["DATA"][0] != "binary":
            raise ValueError(f"binary PCD만 읽음 (DATA {hdr['DATA'][0]}): {path}")
        counts = hdr.get("COUNT", ["1"] * len(hdr["FIELDS"]))
        dt = np.dtype([(name, _NP[(t, int(s))], (int(c),) if int(c) > 1 else ())
                       for name, s, t, c in zip(hdr["FIELDS"], hdr["SIZE"], hdr["TYPE"], counts)])
        a = np.frombuffer(f.read(), dt, count=int(hdr["POINTS"][0]))
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    val = a[field if field in a.dtype.names else "intensity"].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    m = np.isfinite(r) & (r > min_range)
    return xyz[m], val[m]
