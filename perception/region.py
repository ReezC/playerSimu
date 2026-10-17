"""矩形区域（框选）的**一处口径** —— 「区域自动标注」四条链共用 ✓。

⚠⚠ 为什么单独立一个（用户 2026-10-11 ✓ 原话："点击各个标注按钮，**只会在当前帧框选的区域内
  进行自动标注**" ✓）：

  · 区域从**画布框选**来（`gui/canvas.region_picked` ✓ 发的是 `QRectF` ✓ = **画面像素** ✓），
    落到标注链时要变成 `"x,y,w,h"` 这串**字符串** —— `tools/detect_mobs.run_detect` 早就吃它 ✓
    （它还会把检出坐标**偏移回去** ✓ 见那边 `region` 的说明 ✓）；
  · 四条链各有各的出口（模板匹配走 `run_detect` ✓、YOLO 走 `yolo_augment` ✓、
    玩家走 `detect_player.work` ✓）⇒ "**区域外的预测一律丢掉**"这件事必须**同一套判据** ✗
    —— 各写一份的话，边界上必然一个含一个不含 ⇒ **四个按钮给出四种行为** ✓ 那就没法用了 ✓；
  · ⇒ 判据收在这里：**中心点在不在区域内** ✓（`center_inside` ✓）。

⚠ 口径是"**中心点**"而不是"框完全落在里面" ✓（用户要的是"这个区域内标出来的东西" ✓；
  跨边界的框只要它**主要在**区域内就该算 ✓ —— 这也是检测框的常规取舍 ✓）。
"""


def to_spec(rect):
    """`QRect(F)` / `(x, y, w, h)` / 已经是字符串 ⇒ `"x,y,w,h"`（整数 ✓）；空 / 坏 ⇒ `""` ✓。"""
    if rect is None:
        return ""
    if isinstance(rect, str):
        return rect.strip()
    try:
        # QRectF / QRect：都给了 left/top/width/height ✓
        x = int(round(rect.left() if hasattr(rect, "left") else rect[0]))
        y = int(round(rect.top() if hasattr(rect, "top") else rect[1]))
        w = int(round(rect.width() if hasattr(rect, "width") else rect[2]))
        h = int(round(rect.height() if hasattr(rect, "height") else rect[3]))
    except Exception:                            # noqa: BLE001 —— 认不出就当没框 ✓
        return ""
    if w <= 0 or h <= 0:
        return ""
    return "%d,%d,%d,%d" % (x, y, w, h)


def parse(spec):
    """`"x,y,w,h"` ⇒ `(x, y, w, h)`（整数 ✓）；空 / 坏 / 非正 ⇒ `None` ✓。

    ⚠ 与 `detect_mobs.run_detect` 里那段解析**同一口径**（它要 `"x,y,w,h"` ✓ 逗号分隔 ✓）。
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        try:
            v = tuple(int(float(x)) for x in spec)
        except (TypeError, ValueError):
            return None
    else:
        s = str(spec).strip()
        if not s:
            return None
        try:
            v = tuple(int(float(x)) for x in s.split(","))
        except (TypeError, ValueError):
            return None
        if len(v) != 4:
            return None
    return v if v[2] > 0 and v[3] > 0 else None


def center_inside(region, cx, cy):
    """`(cx, cy)`（**画面像素** ✓）的中心点在不在区域里 ⇒ True/False ✓；没区域 ⇒ **True** ✓。

    ⚠ 没给区域 ⇒ 一律 True ✓（= 老行为：整幅画面都算 ✓ 一个字节都不改 ✗ 这正是没框选时该有的样子 ✓）。
    """
    r = parse(region)
    if r is None:
        return True
    x, y, w, h = r
    return (x <= float(cx) <= x + w) and (y <= float(cy) <= y + h)


def filter_norm(region, rows, fw, fh):
    """把**归一化**的框 `[(cls, cx, cy, bw, bh), …]` 里中心不在区域的丢掉 ✓（没区域 ⇒ 原样 ✓）。

    `fw / fh` = 那一帧的宽高（像素 ✓）—— 归一化坐标要乘回来才能跟区域比 ✓。
    """
    r = parse(region)
    if r is None:
        return list(rows)
    if not fw or not fh:
        return list(rows)
    out = []
    for row in rows:
        try:
            cx, cy = float(row[1]) * float(fw), float(row[2]) * float(fh)
        except (TypeError, ValueError, IndexError):
            continue
        if center_inside(r, cx, cy):
            out.append(row)
    return out


def note(region):
    """给日志用的一句人话（没区域 ⇒ `""` ✓）—— "只在这个矩形里标"✓。"""
    r = parse(region)
    if r is None:
        return ""
    return "只标框选区域 (%d,%d) %dx%d" % r
