"""把频道面板放大出来量格子（一次性件）：画上默认目标 (561,391) 的红十字做参照。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import cv2                                            # noqa: E402

from tools.mouse_aim_calib import _connect_input, _grab, _open_stream   # noqa: E402

X0, Y0, X1, Y1 = 480, 330, 920, 560        # 裁剪区（@1366×768 ✓）
SCALE = 2.6

_connect_input()
src, _ = _open_stream()
f = _grab(src)
if f is None:
    print("抓不到帧")
    raise SystemExit(2)
cv2.imwrite("data/_panel_full.png", f)

crop = f[Y0:Y1, X0:X1]
big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
cx, cy = int((561 - X0) * SCALE), int((391 - Y0) * SCALE)
cv2.line(big, (cx, 0), (cx, big.shape[0]), (0, 0, 255), 1)
cv2.line(big, (0, cy), (big.shape[1], cy), (0, 0, 255), 1)
cv2.imwrite("data/_panel_zoom.png", big)
print("裁剪 x %d..%d / y %d..%d，放大 %.1f 倍；红线 = 默认目标 (561,391)" % (X0, X1, Y0, Y1, SCALE))
print("存：data/_panel_zoom.png（%dx%d）" % (big.shape[1], big.shape[0]))
try:
    src.close()
except Exception:                                       # noqa: BLE001
    pass
