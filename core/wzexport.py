"""调用 WzProbe.exe 导出地图清单，并读取结果。

WZ 解析（老版多 wz、0.75 格式）由 C# 的 WzProbe 完成 —— 那套 WzLib 已经
实测能读这套资源（16/17 个 wz 解析成功）。这里只负责：

    1. 找到 WzProbe.exe（可配置，否则自动探测）
    2. 起进程并把它的输出实时转成 ctx.log
    3. 读回生成的 JSON

不重复实现 WZ 解析：一是工作量大，二是已验证的工具没必要重写。
"""

import json
import os
import subprocess
from pathlib import Path

import yaml

from core.context import TaskContext

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "wz.yaml"

DEFAULTS = {
    "wz_dir": r"E:\冒险岛online075\冒险岛online",
    "probe_exe": "",
    "maps_json": "datasets/maps.json",
    "sprite_dir": "datasets/sprites/mob",
}

# 自动探测 WzProbe.exe 的位置（按可能性排序）
EXE_CANDIDATES = (
    r"E:\MyPrograms\RippleRogue\MapleNecrocer\WzProbe\bin\Release\net8.0\WzProbe.exe",
    r"E:\MyPrograms\RippleRogue\MapleNecrocer\WzProbe\bin\Debug\net8.0\WzProbe.exe",
)


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
def load_cfg():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(yaml.safe_load(f) or {})
        except Exception:
            pass
    return cfg


def save_cfg(cfg):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(DEFAULTS)
    merged.update(cfg or {})
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
    return merged


def find_probe_exe(configured=""):
    """找 WzProbe.exe。配置优先，否则探测常见位置。找不到返回空串。"""
    if configured:
        p = Path(configured)
        if p.exists():
            return str(p)
    for c in EXE_CANDIDATES:
        if Path(c).exists():
            return c
    return ""


def maps_json_path(cfg=None):
    cfg = cfg or load_cfg()
    p = Path(cfg.get("maps_json") or DEFAULTS["maps_json"])
    return p if p.is_absolute() else (ROOT / p)


def sprite_dir_path(cfg=None):
    """精灵库目录（绝对路径）。自动标注拿它当模板。"""
    cfg = cfg or load_cfg()
    p = Path(cfg.get("sprite_dir") or DEFAULTS["sprite_dir"])
    return p if p.is_absolute() else (ROOT / p)


# ══════════════════════════════════════════════════════════════
# 读清单
# ══════════════════════════════════════════════════════════════
def load_index(path=None):
    """读地图清单 JSON。不存在或损坏返回 None。"""
    p = Path(path) if path else maps_json_path()
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_maps(only_with_mob=True, keyword=""):
    """把清单整理成下拉列表用的条目。

    筛选分两级：
      1. 精确子串（ID / 地图名 / 地区 / 怪物名 都参与）—— 最优先
      2. 逐字重合度兜底 —— 记错或记不全地图名时还能找到
         实测：把「东郊平原」记成「彩虹村东郊平原」也能命中

    返回 [{id, label, name, region, mobs, mob_names, has_mob, score}, ...]
    """
    idx = load_index()
    if not idx:
        return []

    kw = (keyword or "").strip().lower()
    out = []

    for m in idx.get("maps", []):
        mobs = m.get("mobs") or []
        if only_with_mob and not mobs:
            continue

        mid = m.get("id", "")
        name = (m.get("name") or "").strip()
        region = (m.get("region") or "").strip()
        mob_names = [x for x in (m.get("mob_names") or []) if x]

        score = 1
        if kw:
            score = _score(mid, name, region, mob_names, kw)
            if score <= 0:
                continue

        out.append({
            "id": mid,
            "name": name,
            "region": region,
            "mobs": mobs,
            "mob_names": mob_names,
            "has_mob": bool(mobs),
            "score": score,
            "label": format_label(mid, name, region, mob_names, len(mobs)),
        })

    out.sort(key=lambda x: (-x["score"], x["id"]) if kw else x["id"])
    return out


def _score(mid, name, region, mob_names, kw):
    """匹配分，0 表示不匹配。

    精确子串给高分（关键词越长越优先）；否则按「关键词里有多少字出现在条目里」
    打分，重合度不到 60% 就当没匹配 —— 阈值再低会拖出一大堆无关地图。
    """
    blob = ("%s %s %s %s" % (mid, name, region, " ".join(mob_names))).lower()

    if kw in blob:
        return 1000 + len(kw)

    chars = set(kw)
    if not chars:
        return 0

    hit = sum(1 for c in chars if c in blob)
    if hit < 2:
        # 只命中一个字太容易误伤：搜「野猪」会把所有含「野」的地图全拖出来
        return 0

    ratio = float(hit) / len(chars)
    return int(ratio * 100) if ratio >= 0.5 else 0


def format_label(mid, name, region, mob_names, n_mob):
    """下拉项显示文本。

    本套资源里有 680 张地图在 String.wz 里没有名字，只显示 ID 根本认不出来，
    所以用怪物名来标识 —— 「蜗牛/蓝蜗牛」比「001000000」好认得多。
    """
    title = name or (("(%s)" % region) if region else "(未命名)")
    parts = [mid, title]

    if mob_names:
        shown = "/".join(mob_names[:3])
        if len(mob_names) > 3:
            shown += " 等 %d 种" % len(mob_names)
        parts.append("· " + shown)
    elif n_mob:
        parts.append("· %d 种怪" % n_mob)

    return "   ".join(parts)


def build_mob_name_map():
    """从地图清单收集「怪 id → 名称」映射（同名取第一个非空值）。

    精灵库的 meta.tsv 里没有怪物名，名字只散落在 maps.json 各条地图记录里，
    这里统一收一遍，供「确认要识别怪物」弹窗显示用。
    """
    idx = load_index()
    name_map = {}
    if not idx:
        return name_map
    for m in idx.get("maps", []):
        mobs = m.get("mobs") or []
        names = m.get("mob_names") or []
        for i, mid in enumerate(mobs):
            if mid not in name_map and i < len(names) and names[i]:
                name_map[mid] = names[i]
    return name_map


def list_sprite_mobs():
    """列出精灵库里所有有 stand 帧的怪 id（排序，供手动选怪用）。"""
    root = sprite_dir_path()
    out = []
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and list(d.glob("stand_*.png")):
                out.append(d.name)
    return sorted(out)


# ══════════════════════════════════════════════════════════════
# 导出
# ══════════════════════════════════════════════════════════════
def run_export_maps(params, ctx=None):
    """导出地图清单。params: exe / wz_dir / out

    起 WzProbe.exe 子进程，把它的 stdout 逐行转成日志 ——
    导出要跑几秒到几分钟，不能让界面看起来卡死。
    """
    ctx = ctx or TaskContext()

    cfg = load_cfg()
    exe = find_probe_exe(params.get("exe") or cfg.get("probe_exe") or "")
    wz_dir = params.get("wz_dir") or cfg.get("wz_dir") or ""
    out = params.get("out") or str(maps_json_path(cfg))

    if not exe:
        raise FileNotFoundError(
            "找不到 WzProbe.exe。\n"
            "请先编译:\n"
            "  cd E:\\MyPrograms\\RippleRogue\\MapleNecrocer\n"
            "  dotnet build WzProbe/WzProbe.csproj -c Release\n"
            "然后在导出对话框里指定它的路径。")

    if not Path(exe).exists():
        raise FileNotFoundError("WzProbe.exe 不存在: %s" % exe)

    # 空串必须单独判：Path("") 在 Windows 上等于 Path(".")，
    # is_dir() 会返回 True，于是当前目录被当成 WZ 目录，报出一堆莫名其妙的错。
    if not wz_dir:
        raise ValueError("没有指定 WZ 目录（在导出对话框里设置，或写进 config/wz.yaml）")

    if not Path(wz_dir).is_dir():
        raise FileNotFoundError("WZ 目录不存在: %s" % wz_dir)

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    ctx.log("WzProbe : %s" % exe)
    ctx.log("WZ 目录 : %s" % wz_dir)
    ctx.log("输出    : %s" % out)
    ctx.log("开始解析 Map.wz（地图多，可能要几分钟）…")

    cmd = [exe, "dump-map", wz_dir, out]

    # Windows 下不弹黑框
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",      # WzProbe 里已设 Console.OutputEncoding = UTF8
        errors="replace",
        bufsize=1,
        creationflags=flags,
    )

    canceled = False
    try:
        for line in proc.stdout:
            if ctx.canceled():
                canceled = True
                proc.kill()
                break
            line = line.rstrip()
            if line:
                ctx.log(line)
                if "[" in line and "/" in line:
                    # 形如 "  [200/3733]  有怪 134 ..." —— 折算成进度
                    try:
                        seg = line.strip().split("]")[0].strip("[")
                        cur, total = seg.split("/")
                        ctx.progress(int(cur), int(total), "解析地图")
                    except Exception:
                        pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()

    if canceled:
        ctx.log("已取消导出", "warn")
        return {"summary": "已取消"}

    if proc.returncode != 0:
        raise RuntimeError("WzProbe 退出码 %d，导出失败" % proc.returncode)

    idx = load_index(out)
    if not idx:
        raise RuntimeError("导出结束但读不到清单文件: %s" % out)

    total = idx.get("count", 0)
    with_mob = idx.get("with_mob", 0)
    named = sum(1 for m in idx.get("maps", []) if (m.get("name") or "").strip())

    summary = "地图 %d 张 / 有怪 %d / 有名字 %d" % (total, with_mob, named)

    ctx.log("── 导出完成 ──", "ok")
    ctx.log("  地图总数 %d" % total)
    ctx.log("  其中有怪 %d" % with_mob)
    ctx.log("  有名字   %d" % named)
    ctx.log("  清单文件 %s" % out)

    ctx.progress(total, total, "完成")
    return {"summary": summary, "index": idx, "out": out}
