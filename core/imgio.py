"""OpenCV 读写封装，兼容中文路径。

**为什么必须走这里**：cv2.imread / cv2.imwrite 内部用 C 的 fopen，
遇到非 ASCII 路径会**静默失败** —— 读返回 None、写返回 False，都不抛异常。

而项目目录名常常是中文地图名（比如「彩虹村东郊平原」），
一旦漏掉这层封装，表现就是「自动标注 0 检出」这种查不出来的问题。

原来只有写入侧做了兼容，读取侧漏了，结果在所有中文项目上全军覆没。
"""

from pathlib import Path

# ⚠ **可选依赖**（同 `core/wincap.py` 顶部那条说明 ✓）：A 机部署台
#   （只列窗口 / 起 ffmpeg）**不需要 opencv** ✓ —— 这里顶层硬导入会让
#   `import core.wincap` 连带把 `imgio` 一起拽进来 ⇒ `No module named 'cv2'` ✗
#   （用户 2026-10-17 现场 ✓）。真读写图片时才要求它 ✓。
try:
    import cv2
except ImportError:                        # noqa: BLE001
    cv2 = None
try:
    import numpy as np
except ImportError:                        # noqa: BLE001
    np = None


def _need_cv(where):
    """读/写图片前确认 opencv 在；缺了就**说清怎么装** ✓（不许 AttributeError ✗）。"""
    if cv2 is not None and np is not None:
        return
    raise RuntimeError("%s 需要 opencv-python + numpy（pip install opencv-python numpy）"
                       % where)


def imread(path, flags=None):
    """读图（默认 `cv2.IMREAD_COLOR` ✓ —— 延迟取，因为 `cv2` 可能是 None ✓）。"""
    _need_cv("imread")
    if flags is None:
        flags = cv2.IMREAD_COLOR       # 中文路径 ⇒ 走 imdecode（见文件头说明 ✓）
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
    _need_cv("imwrite")
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
