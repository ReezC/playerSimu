"""OpenCV 读写封装，兼容中文路径。

**为什么必须走这里**：cv2.imread / cv2.imwrite 内部用 C 的 fopen，
遇到非 ASCII 路径会**静默失败** —— 读返回 None、写返回 False，都不抛异常。

而项目目录名常常是中文地图名（比如「彩虹村东郊平原」），
一旦漏掉这层封装，表现就是「自动标注 0 检出」这种查不出来的问题。

原来只有写入侧做了兼容，读取侧漏了，结果在所有中文项目上全军覆没。
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图，支持中文路径。失败返回 None（和 cv2.imread 行为一致）。"""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None

    if not data:
        return None

    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def imwrite(path, img, quality=None):
    """写图，支持中文路径。返回 True 表示成功。

    quality 只在 jpg 时生效。
    """
    p = Path(path)

    params = []
    if quality is not None and p.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    ok, buf = cv2.imencode(p.suffix or ".png", img, params)
    if not ok:
        raise IOError("图像编码失败: %s" % path)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(buf.tobytes())
    return True
