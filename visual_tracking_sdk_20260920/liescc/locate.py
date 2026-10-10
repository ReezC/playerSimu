"""框体定位: 从游戏画面找出视觉任务弹窗的**内容区** rect (x, y, w, h)。

## 依据(实测 lies/startlies2.png)

- 图 1359x1061, **唯一透明矩形孔 = (13, 57, 1333x888)**; 环带 上57/下116/左13/右13
- 孔外与 startlies.png 逐字节相同 → 环带是稳定的外观特征
- 由此得到与分辨率无关的比例常量(见 HOLE_*)

## 定位路径

主程序使用顶部标题条与底部提示栏的双锚点共识；候选在20个处理帧内累计3次
稳定确认后直接固定本轮ROI，随后由YOLO负责图形检测和追踪。
旧几何法和整框模板法继续保留给离线工具与兼容接口，但不再能直接取得实战鼠标控制权。
整框模板仍使用 TM_CCORR_NORMED + mask，避免 TM_SQDIFF_NORMED 被均匀背景欺骗。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# startlies2.png 的孔位比例(相对整图宽高)
HOLE_RX = 13 / 1359
HOLE_RY = 57 / 1061
HOLE_RW = 1333 / 1359
HOLE_RH = 888 / 1061

# 环带 = 近白低饱和(实测: 视频里环带 BGR≈(246,243,244), startlies2.png 同)。
# 早先按固定色 (219,210,202)±42 判定, 在真实素材上只命中约一半环带像素 —— 那个均值
# 掺进了边框上的暗色装饰纹。改成"够亮 + 通道差够小", 模板与视频两边都是干净的 1.0/0.0。
RING_MIN = 195          # min(B,G,R) 下限
RING_SPREAD = 45        # max-min 上限(白色三通道接近)
CONTENT_BGR = (74, 130, 159)   # 内部石纹参考色(验证用)
CONTENT_TOL = 60

MIN_AREA_FRAC = 0.05      # 弹窗至少占画面这么大
ASPECT = 1359 / 1061
ASPECT_TOL = 0.35
SNAP_RANGE = 30           # 边界吸附搜索半径(px)
SNAP_RING_FRAC = 0.5      # 一行/列中环色占比低于此值即视为已进入内容区

_TEMPLATE = Path(__file__).resolve().parent / "startlies2.png"
TITLE_HEIGHT = 57
TEMPLATE_WIDTH = 1359
TEMPLATE_HEIGHT = 1061
FOOTER_Y = 945
TITLE_MATCH_DOWNSCALE = 0.25
TITLE_MATCH_THRESHOLD = 0.88

# 主程序实战定位使用的双锚点参数。两个锚点不要求像素级一致：比例取整、
# DPI缩放和压缩抖动都有明确冗余；主程序再以20个处理帧内累计3次稳定确认
# 固定本轮ROI。geometry YOLO只负责随后追踪，不参与远程ROI晋升。
ANCHOR_MATCH_DOWNSCALE = 0.25
ANCHOR_MIN_SCALE = 0.20
ANCHOR_MAX_SCALE = 2.50
ANCHOR_SCALE_STEP = 0.025
TITLE_ANCHOR_THRESHOLD = 0.86
FOOTER_ANCHOR_THRESHOLD = 0.70
ANCHOR_SCALE_TOLERANCE = 0.08
ANCHOR_EDGE_TOLERANCE_FRAC = 0.05
ANCHOR_VERTICAL_TOLERANCE_FRAC = 0.08


@dataclass(frozen=True)
class AnchorMatch:
    x: float
    y: float
    width: float
    height: float
    scale: float
    score: float


@dataclass(frozen=True)
class LieWindowCandidate:
    outer_rect: tuple[int, int, int, int]
    content_rect: tuple[int, int, int, int]
    title_score: float
    footer_score: float
    scale: float


@dataclass(frozen=True)
class LocateDiagnostics:
    reason: str
    title_best_score: float = -1.0
    footer_best_score: float = -1.0
    title_matches: int = 0
    footer_matches: int = 0
    paired_candidates: int = 0
    reject_counts: tuple[tuple[str, int], ...] = ()


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """中文路径安全读图(项目约定: cv2.imdecode + np.fromfile)。"""
    p = Path(path)
    if not p.exists():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _ring_mask(bgr: np.ndarray) -> np.ndarray:
    mn = bgr.min(axis=2).astype(np.int16)
    mx = bgr.max(axis=2).astype(np.int16)
    return (((mn > RING_MIN) & ((mx - mn) < RING_SPREAD)).astype(np.uint8)) * 255


@lru_cache(maxsize=1)
def _title_template_gray() -> np.ndarray | None:
    """读取稳定的窗体标题条；透明内容区和动态几何图形不参与兜底。"""
    rgba = imread_unicode(_TEMPLATE, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
        return None
    title = rgba[:min(TITLE_HEIGHT, rgba.shape[0]), :, :3]
    if title.size == 0:
        return None
    return cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)


@lru_cache(maxsize=1)
def _anchor_templates_gray() -> tuple[np.ndarray, np.ndarray] | None:
    """Return the two stable opaque bands surrounding the dynamic content."""
    rgba = imread_unicode(_TEMPLATE, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
        return None
    if rgba.shape[0] <= FOOTER_Y or rgba.shape[1] < 32:
        return None
    title = rgba[:TITLE_HEIGHT, :, :3]
    footer = rgba[FOOTER_Y:, :, :3]
    if title.size == 0 or footer.size == 0:
        return None
    return (
        cv2.cvtColor(title, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(footer, cv2.COLOR_BGR2GRAY),
    )


@lru_cache(maxsize=256)
def _resized_anchor_template(
    anchor_index: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Cache the stable anchor pyramid shared by consecutive locator frames."""
    templates = _anchor_templates_gray()
    if templates is None or anchor_index not in (0, 1):
        return None
    return cv2.resize(
        templates[anchor_index],
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    )


def _resize_match_template(
    template: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Reuse title/footer scales without changing generic test/tool inputs."""
    templates = _anchor_templates_gray()
    if templates is not None:
        for index, anchor in enumerate(templates):
            if template is anchor:
                cached = _resized_anchor_template(index, width, height)
                if cached is not None:
                    return cached
    return cv2.resize(
        template,
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    )


def _scale_values(frame_shape: tuple[int, ...]) -> tuple[float, ...]:
    frame_height, frame_width = frame_shape[:2]
    min_by_frame = max(
        ANCHOR_MIN_SCALE,
        frame_width * 0.20 / TEMPLATE_WIDTH,
        frame_height * 0.20 / TEMPLATE_HEIGHT,
    )
    max_by_frame = min(
        frame_width * 0.95 / TEMPLATE_WIDTH,
        frame_height * 0.95 / TEMPLATE_HEIGHT,
        ANCHOR_MAX_SCALE,
    )
    if max_by_frame < min_by_frame:
        return ()
    first = math.ceil(min_by_frame / ANCHOR_SCALE_STEP) * ANCHOR_SCALE_STEP
    count = int(math.floor((max_by_frame - first) / ANCHOR_SCALE_STEP))
    return tuple(
        round(first + index * ANCHOR_SCALE_STEP, 4)
        for index in range(count + 1)
    )


def _collect_anchor_matches(
    frame_gray: np.ndarray,
    template: np.ndarray,
    *,
    min_score: float,
    max_matches: int = 12,
    scales: tuple[float, ...] | None = None,
    downscale: float | None = None,
    refine: bool = True,
    diagnostics: dict[str, float | int] | None = None,
) -> list[AnchorMatch]:
    """Collect peaks with coarse search plus local scale refinement."""
    down = float(
        downscale
        if downscale is not None
        else (0.50 if min(frame_gray.shape[:2]) <= 800 else ANCHOR_MATCH_DOWNSCALE)
    )
    small = cv2.resize(
        frame_gray,
        None,
        fx=down,
        fy=down,
        interpolation=cv2.INTER_AREA,
    )
    small_height, small_width = small.shape[:2]
    coarse_scales = scales or _scale_values(frame_gray.shape)

    def scan(
        values: tuple[float, ...], threshold: float
    ) -> list[AnchorMatch]:
        found: list[AnchorMatch] = []
        for scale in values:
            width = int(round(template.shape[1] * scale * down))
            height = int(round(template.shape[0] * scale * down))
            if width < 24 or height < 4 or width > small_width or height > small_height:
                continue
            resized = _resize_match_template(template, width, height)
            scores = cv2.matchTemplate(small, resized, cv2.TM_CCOEFF_NORMED)
            scores[~np.isfinite(scores)] = -1.0
            _, score, _, location = cv2.minMaxLoc(scores)
            if diagnostics is not None:
                diagnostics["best_score"] = max(
                    float(diagnostics.get("best_score", -1.0)), float(score)
                )
            if float(score) < float(threshold):
                continue
            found.append(
                AnchorMatch(
                    x=location[0] / down,
                    y=location[1] / down,
                    width=width / down,
                    height=height / down,
                    scale=float(scale),
                    score=float(score),
                )
            )
        return found

    coarse = scan(tuple(coarse_scales), max(-1.0, min_score - 0.10))
    refine_scales = (
        {
            round(match.scale + offset, 4)
            for match in sorted(coarse, key=lambda item: item.score, reverse=True)[:6]
            for offset in (-ANCHOR_SCALE_STEP / 2.0, ANCHOR_SCALE_STEP / 2.0)
            if ANCHOR_MIN_SCALE <= match.scale + offset <= ANCHOR_MAX_SCALE
        }
        if refine
        else set()
    )
    refined = scan(tuple(sorted(refine_scales)), min_score) if refine_scales else []
    matches = [match for match in coarse if match.score >= min_score] + refined

    # A true anchor often wins at several adjacent scales. Keep those peaks from
    # crowding out a second spatial candidate while retaining scale redundancy.
    kept: list[AnchorMatch] = []
    for item in sorted(matches, key=lambda match: match.score, reverse=True):
        duplicate = any(
            abs(item.x - other.x) <= max(6.0, item.width * 0.03)
            and abs(item.y - other.y) <= max(4.0, item.height * 0.60)
            and abs(item.scale - other.scale) <= ANCHOR_SCALE_STEP * 1.1
            for other in kept
        )
        if not duplicate:
            kept.append(item)
        if len(kept) >= max_matches:
            break
    if diagnostics is not None:
        diagnostics["matches"] = len(kept)
    return kept


def _pair_anchor_matches(
    title_matches: list[AnchorMatch],
    footer_matches: list[AnchorMatch],
    frame_shape: tuple[int, ...],
    reject_counts: dict[str, int] | None = None,
) -> list[LieWindowCandidate]:
    """Pair title/footer anchors with tolerant scale, edge and layout gates."""
    frame_height, frame_width = frame_shape[:2]
    candidates: list[LieWindowCandidate] = []

    def rejected(reason: str) -> None:
        if reject_counts is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    for title in title_matches:
        for footer in footer_matches:
            mean_scale = max(1e-6, (title.scale + footer.scale) / 2.0)
            relative_scale_error = abs(title.scale - footer.scale) / mean_scale
            if relative_scale_error > ANCHOR_SCALE_TOLERANCE:
                rejected("SCALE_MISMATCH")
                continue

            expected_width = TEMPLATE_WIDTH * mean_scale
            edge_tolerance = max(8.0, expected_width * ANCHOR_EDGE_TOLERANCE_FRAC)
            title_right = title.x + title.width
            footer_right = footer.x + footer.width
            if abs(title.x - footer.x) > edge_tolerance:
                rejected("LEFT_EDGE_MISMATCH")
                continue
            if abs(title_right - footer_right) > edge_tolerance:
                rejected("RIGHT_EDGE_MISMATCH")
                continue

            expected_footer_y = title.y + FOOTER_Y * mean_scale
            expected_height = TEMPLATE_HEIGHT * mean_scale
            vertical_tolerance = max(
                10.0, expected_height * ANCHOR_VERTICAL_TOLERANCE_FRAC
            )
            if abs(footer.y - expected_footer_y) > vertical_tolerance:
                rejected("LAYOUT_MISMATCH")
                continue

            top = title.y
            layout_scale = max(1e-6, (footer.y - title.y) / FOOTER_Y)
            observed_height = TEMPLATE_HEIGHT * layout_scale
            # The footer has far more stable vertical texture than the shallow
            # title strip. The distance between both anchors is the most reliable
            # scale signal; footer matching supplies the horizontal centre.
            outer_width = TEMPLATE_WIDTH * layout_scale
            center_x = footer.x + footer.width / 2.0
            left = center_x - outer_width / 2.0
            outer = (
                int(round(left)),
                int(round(top)),
                int(round(outer_width)),
                int(round(observed_height)),
            )
            x, y, width, height = outer
            if width < 160 or height < 120:
                rejected("WINDOW_TOO_SMALL")
                continue
            if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
                rejected("WINDOW_OUT_OF_BOUNDS")
                continue
            # A lie-detector window is a substantial centered modal. This broad
            # gate removes tiny top-left UI elements without assuming one resolution.
            center_x = x + width / 2.0
            center_y = y + height / 2.0
            if not (frame_width * 0.20 <= center_x <= frame_width * 0.80):
                rejected("HORIZONTAL_CENTER_MISMATCH")
                continue
            if not (frame_height * 0.15 <= center_y <= frame_height * 0.88):
                rejected("VERTICAL_CENTER_MISMATCH")
                continue

            content = _content_rect_from_outer(*outer)
            candidates.append(
                LieWindowCandidate(
                    outer_rect=outer,
                    content_rect=content,
                    title_score=title.score,
                    footer_score=footer.score,
                    scale=layout_scale,
                )
            )
    return sorted(
        candidates,
        key=lambda item: (item.title_score + item.footer_score) / 2.0,
        reverse=True,
    )


def locate_by_dual_anchor_diagnostic(
    frame_bgr: np.ndarray,
) -> tuple[LieWindowCandidate | None, LocateDiagnostics]:
    """Locate a modal and retain the strongest rejection evidence."""
    templates = _anchor_templates_gray()
    if (
        templates is None
        or frame_bgr is None
        or frame_bgr.size == 0
        or frame_bgr.ndim not in (2, 3)
    ):
        reason = "TEMPLATE_MISSING" if templates is None else "INVALID_FRAME"
        return None, LocateDiagnostics(reason=reason)
    gray = (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if frame_bgr.ndim == 3
        else frame_bgr
    )
    title_template, footer_template = templates
    title_stats: dict[str, float | int] = {}
    titles = _collect_anchor_matches(
        gray,
        title_template,
        min_score=TITLE_ANCHOR_THRESHOLD,
        diagnostics=title_stats,
    )
    if not titles:
        return None, LocateDiagnostics(
            reason="TITLE_MISS",
            title_best_score=float(title_stats.get("best_score", -1.0)),
        )
    footer_stats: dict[str, float | int] = {}
    footers = _collect_anchor_matches(
        gray,
        footer_template,
        min_score=FOOTER_ANCHOR_THRESHOLD,
        diagnostics=footer_stats,
    )
    if not footers:
        return None, LocateDiagnostics(
            reason="FOOTER_MISS",
            title_best_score=float(title_stats.get("best_score", -1.0)),
            footer_best_score=float(footer_stats.get("best_score", -1.0)),
            title_matches=len(titles),
        )
    reject_counts: dict[str, int] = {}
    paired = _pair_anchor_matches(
        titles, footers, frame_bgr.shape, reject_counts=reject_counts
    )
    reason = "MATCHED"
    if not paired:
        reason = (
            max(reject_counts, key=reject_counts.get)
            if reject_counts
            else "PAIR_MISMATCH"
        )
    diagnostics = LocateDiagnostics(
        reason=reason,
        title_best_score=float(title_stats.get("best_score", -1.0)),
        footer_best_score=float(footer_stats.get("best_score", -1.0)),
        title_matches=len(titles),
        footer_matches=len(footers),
        paired_candidates=len(paired),
        reject_counts=tuple(sorted(reject_counts.items())),
    )
    return (paired[0] if paired else None), diagnostics


def locate_by_dual_anchor(frame_bgr: np.ndarray) -> LieWindowCandidate | None:
    """Locate a modal only when its title and footer independently agree."""
    return locate_by_dual_anchor_diagnostic(frame_bgr)[0]


def _rects_are_close(
    previous: tuple[int, int, int, int],
    current: tuple[int, int, int, int],
) -> bool:
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    if min(pw, ph, cw, ch) <= 0:
        return False
    previous_center = (px + pw / 2.0, py + ph / 2.0)
    current_center = (cx + cw / 2.0, cy + ch / 2.0)
    center_distance = math.hypot(
        previous_center[0] - current_center[0],
        previous_center[1] - current_center[1],
    )
    center_tolerance = max(4.0, math.hypot(pw, ph) * 0.025)
    width_change = abs(cw - pw) / max(pw, cw)
    height_change = abs(ch - ph) / max(ph, ch)
    return (
        center_distance <= center_tolerance
        and width_change <= 0.08
        and height_change <= 0.08
    )


def candidates_are_stable(
    previous: LieWindowCandidate | None,
    current: LieWindowCandidate | None,
) -> bool:
    if previous is None or current is None:
        return False
    return _rects_are_close(previous.outer_rect, current.outer_rect)


def validate_dual_anchor_near(
    frame_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
) -> LieWindowCandidate | None:
    """Revalidate a locked content rect near its current modal location."""
    outer_width = rect[2] / HOLE_RW
    outer_height = rect[3] / HOLE_RH
    outer = (
        int(round(rect[0] - outer_width * HOLE_RX)),
        int(round(rect[1] - outer_height * HOLE_RY)),
        int(round(outer_width)),
        int(round(outer_height)),
    )
    margin_x = max(12, int(round(outer[2] * 0.12)))
    margin_y = max(12, int(round(outer[3] * 0.12)))
    x0 = max(0, outer[0] - margin_x)
    y0 = max(0, outer[1] - margin_y)
    x1 = min(frame_bgr.shape[1], outer[0] + outer[2] + margin_x)
    y1 = min(frame_bgr.shape[0], outer[1] + outer[3] + margin_y)
    if x1 - x0 < 160 or y1 - y0 < 120:
        return None
    local_frame = frame_bgr[y0:y1, x0:x1]
    gray = (
        cv2.cvtColor(local_frame, cv2.COLOR_BGR2GRAY)
        if local_frame.ndim == 3
        else local_frame
    )
    templates = _anchor_templates_gray()
    if templates is None:
        return None
    expected_scale = outer[3] / TEMPLATE_HEIGHT
    narrow_scales = tuple(
        expected_scale * factor for factor in (0.92, 0.96, 1.00, 1.04, 1.08)
    )
    title_template, footer_template = templates
    titles = _collect_anchor_matches(
        gray,
        title_template,
        min_score=TITLE_ANCHOR_THRESHOLD,
        max_matches=6,
        scales=narrow_scales,
        downscale=ANCHOR_MATCH_DOWNSCALE,
        refine=False,
    )
    footers = _collect_anchor_matches(
        gray,
        footer_template,
        min_score=FOOTER_ANCHOR_THRESHOLD,
        max_matches=6,
        scales=narrow_scales,
        downscale=ANCHOR_MATCH_DOWNSCALE,
        refine=False,
    )
    paired = _pair_anchor_matches(titles, footers, local_frame.shape)
    if not paired:
        return None
    local = paired[0]
    lx, ly, lw, lh = local.outer_rect
    translated_outer = (lx + x0, ly + y0, lw, lh)
    if not _rects_are_close(outer, translated_outer):
        return None
    cx, cy, cw, ch = local.content_rect
    return LieWindowCandidate(
        outer_rect=translated_outer,
        content_rect=(cx + x0, cy + y0, cw, ch),
        title_score=local.title_score,
        footer_score=local.footer_score,
        scale=local.scale,
    )


def lie_window_title_present(
    frame_bgr: np.ndarray,
    *,
    min_score: float = TITLE_MATCH_THRESHOLD,
) -> bool:
    """检测视觉任务窗体标题条，供离线诊断或显式调用方使用。

    该函数不参与主程序提示入口：主程序只允许 ``star1/2/3`` 开启提示，
    接管后再由 ``resolve_rect``/双锚点流程完整定位并验证内容区。近似标题
    匹配结果本身不能用于启动视觉任务会话。
    """
    template = _title_template_gray()
    if (
        template is None
        or frame_bgr is None
        or frame_bgr.size == 0
        or frame_bgr.ndim not in (2, 3)
    ):
        return False
    gray = (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if frame_bgr.ndim == 3
        else frame_bgr
    )
    small = cv2.resize(
        gray,
        None,
        fx=TITLE_MATCH_DOWNSCALE,
        fy=TITLE_MATCH_DOWNSCALE,
        interpolation=cv2.INTER_AREA,
    )
    frame_height, frame_width = small.shape[:2]
    # 从1:1附近向两侧搜索，常见分辨率通常前几次即可命中。
    scales = [1.0]
    for index in range(1, 31):
        delta = index * 0.025
        scales.extend((1.0 - delta, 1.0 + delta))
    for scale in scales:
        if not 0.25 <= scale <= 1.50:
            continue
        width = int(round(template.shape[1] * scale * TITLE_MATCH_DOWNSCALE))
        height = int(round(template.shape[0] * scale * TITLE_MATCH_DOWNSCALE))
        if width < 24 or height < 4 or width > frame_width or height > frame_height:
            continue
        resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        scores = cv2.matchTemplate(small, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(scores)
        if float(score) >= float(min_score):
            return True
    return False


def _content_rect_from_outer(x: int, y: int, w: int, h: int
                             ) -> tuple[int, int, int, int]:
    return (int(round(x + w * HOLE_RX)), int(round(y + h * HOLE_RY)),
            int(round(w * HOLE_RW)), int(round(h * HOLE_RH)))


# ---------------- 几何定位(主路径) ----------------
def locate_by_geometry(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """亮环连通域 → 外框 bbox → 比例换算内容区 → 边界吸附。失败返回 None。"""
    h, w = frame_bgr.shape[:2]
    mask = _ring_mask(frame_bgr)
    # 环带可能被抗锯齿/波浪打断, 先闭运算连起来
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for i in range(1, n):
        bx, by, bw, bh, _ = stats[i]
        if bw * bh < MIN_AREA_FRAC * w * h:
            continue
        if bh <= 0 or abs(bw / bh - ASPECT) > ASPECT_TOL:
            continue
        if bw * bh > best_area:
            best_area = bw * bh
            best = (int(bx), int(by), int(bw), int(bh))
    if best is None:
        return None
    return _snap_content(frame_bgr, _content_rect_from_outer(*best))


def _snap_content(frame_bgr: np.ndarray, rect: tuple[int, int, int, int]
                  ) -> tuple[int, int, int, int]:
    """把内容区四边吸附到"环色→石纹"的颜色跃变处。"""
    fh, fw = frame_bgr.shape[:2]
    x, y, w, h = rect
    mask = _ring_mask(frame_bgr)

    def ring_frac_row(yy: int) -> float:
        if not (0 <= yy < fh):
            return 1.0
        seg = mask[yy, max(0, x):min(fw, x + w)]
        return float(np.count_nonzero(seg)) / max(1, seg.size)

    def ring_frac_col(xx: int) -> float:
        if not (0 <= xx < fw):
            return 1.0
        seg = mask[max(0, y):min(fh, y + h), xx]
        return float(np.count_nonzero(seg)) / max(1, seg.size)

    top = _scan(lambda d: ring_frac_row(y + d))
    bot = _scan(lambda d: ring_frac_row(y + h - 1 - d))
    left = _scan(lambda d: ring_frac_col(x + d))
    right = _scan(lambda d: ring_frac_col(x + w - 1 - d))

    nx = max(0, x + left)
    ny = max(0, y + top)
    nw = min(fw - nx, w - left - right)
    nh = min(fh - ny, h - top - bot)
    if nw < 32 or nh < 32:          # 吸附失败(整块都是环色?) → 保留原 rect
        return rect
    return (nx, ny, nw, nh)


def _scan(frac_at) -> int:
    """从边界向内找第一处环色占比 < SNAP_RING_FRAC 的偏移; 找不到返回 0。"""
    for d in range(0, SNAP_RANGE + 1):
        if frac_at(d) < SNAP_RING_FRAC:
            return d
    return 0


# ---------------- 模板定位(备用) ----------------
def locate_by_template(frame_bgr: np.ndarray, tmpl_path: str | Path | None = None,
                       scales: tuple[float, ...] | None = None,
                       min_score: float = 0.80) -> tuple[int, int, int, int] | None:
    """环带模板 + mask 多尺度 TM_CCORR_NORMED(越大越好)。在降采样帧上搜索。"""
    rgba = imread_unicode(tmpl_path or _TEMPLATE, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return None
    tmpl = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    ring = (alpha > 0).astype(np.uint8) * 255      # 不透明环带才参与匹配

    down = 0.25
    small = cv2.resize(frame_bgr, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    if scales is None:
        scales = tuple(round(0.50 + 0.05 * i, 2) for i in range(21))  # 0.50~1.50

    best: tuple[int, int, int, int] | None = None
    best_score = -1.0
    for s in scales:
        tw = int(round(tmpl.shape[1] * s * down))
        th = int(round(tmpl.shape[0] * s * down))
        if tw < 24 or th < 24 or tw > sw or th > sh:
            continue
        t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        m = cv2.resize(ring, (tw, th), interpolation=cv2.INTER_NEAREST)
        try:
            res = cv2.matchTemplate(small, t, cv2.TM_CCORR_NORMED, mask=m)
        except cv2.error:
            continue
        res[~np.isfinite(res)] = -1.0      # 带 mask 时可能出现 inf/nan
        _, mx, _, mloc = cv2.minMaxLoc(res)
        if mx > best_score:
            best_score = float(mx)
            best = (int(mloc[0] / down), int(mloc[1] / down),
                    int(round(tmpl.shape[1] * s)), int(round(tmpl.shape[0] * s)))
    if best is None or best_score < min_score:
        return None
    return _snap_content(frame_bgr, _content_rect_from_outer(*best))


# ---------------- 统一入口 ----------------
def validate_rect(frame_bgr: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    """校验 rect 真的是内容区。**没有这一步会缓存误检**。

    实测踩过的坑: 视觉任务弹窗还没出现的那一帧, ``locate_by_template`` 在
    TM_CCORR_NORMED 下给出 0.8+ 的假匹配 (439,435,733,489), 还被写进了缓存, 之后
    全程用错 rect。所以模板/几何的结果都必须过这一关才准返回和缓存。

    两个条件:
    1. 内部主要是石纹色(排除"贴在均匀亮背景上"的假匹配)
    2. 紧贴外侧的四条带里, **至少两条**是环带色(允许弹窗被画面边缘裁掉一两边)
    """
    x, y, w, h = rect
    fh, fw = frame_bgr.shape[:2]
    if w < 32 or h < 32 or x < 0 or y < 0 or x + w > fw or y + h > fh:
        return False

    inner = frame_bgr[y + h // 4:y + h - h // 4, x + w // 4:x + w - w // 4]
    if inner.size == 0:
        return False
    d = np.abs(inner.astype(np.int16) - np.asarray(CONTENT_BGR, np.int16))
    stone = float(np.count_nonzero(d.max(axis=2) < CONTENT_TOL)) / max(1, inner[:, :, 0].size)
    if stone < 0.25:
        return False

    band = max(3, min(10, min(w, h) // 40))
    mask = _ring_mask(frame_bgr)
    sides = (
        mask[max(0, y - band):y, x:x + w],              # 上
        mask[y + h:min(fh, y + h + band), x:x + w],     # 下
        mask[y:y + h, max(0, x - band):x],              # 左
        mask[y:y + h, x + w:min(fw, x + w + band)],     # 右
    )
    ok_sides = sum(1 for s in sides
                   if s.size and float(np.count_nonzero(s)) / s.size > 0.5)
    return ok_sides >= 2


def parse_rect(text: str) -> tuple[int, int, int, int]:
    """解析 --rect x,y,w,h。"""
    parts = [int(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"rect 需要 x,y,w,h 四个正整数: {text!r}")
    return (parts[0], parts[1], parts[2], parts[3])


def resolve_rect(frame_bgr: np.ndarray,
                 override: tuple[int, int, int, int] | None = None,
                 cache_path: str | Path | None = None,
                 use_template: bool = True) -> tuple[int, int, int, int] | None:
    """优先级: 手动覆盖 > 缓存 > 几何 > 模板。命中后写缓存。

    缓存格式(``out/rect_cache.json``): ``{"<画面宽>x<画面高>": [x, y, w, h]}``
    """
    if override is not None:
        return override

    h, w = frame_bgr.shape[:2]
    key = f"{w}x{h}"
    cache = _load_cache(cache_path)
    hit = cache.get(key)
    if isinstance(hit, list) and len(hit) == 4:
        return (int(hit[0]), int(hit[1]), int(hit[2]), int(hit[3]))

    rect = locate_by_geometry(frame_bgr)
    if rect is not None and not validate_rect(frame_bgr, rect):
        rect = None
    if rect is None and use_template:
        rect = locate_by_template(frame_bgr)
        if rect is not None and not validate_rect(frame_bgr, rect):
            rect = None
    if rect is not None:
        cache[key] = list(rect)
        _save_cache(cache_path, cache)
    return rect


def _load_cache(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"[liescc] rect 缓存读取失败, 忽略: {e}")
        return {}


def _save_cache(path: str | Path | None, cache: dict) -> None:
    if path is None:
        return
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except OSError as e:
        print(f"[liescc] rect 缓存写入失败: {e}")
