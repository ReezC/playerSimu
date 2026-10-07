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
    # ⭐ 掉落物（道具）图库（用户 2026-10-04 ✓ 见下面"掉落物图库"那一段的说明）——
    #   ⚠ 这两个**现在都还不存在**（外部 WzProbe 还没有道具子命令 ✗）⇒ 界面会如实说"还没导出" ✓
    "drop_sprite_dir": "datasets/sprites/drop",
    "drops_json": "datasets/drops.json",
}

# 自动探测 WzProbe.exe 的位置（按可能性排序）
EXE_CANDIDATES = (
    r"E:\MyPrograms\RippleRogue\MapleNecrocer\WzProbe\bin\Release\net8.0\WzProbe.exe",
    r"E:\MyPrograms\RippleRogue\MapleNecrocer\WzProbe\bin\Debug\net8.0\WzProbe.exe",
)


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
#: `load_cfg` 的缓存：`{"stamp": 文件指纹, "cfg": {...}}`（见下面那段说明 ✓）。
_CFG_CACHE = {}


def load_cfg():
    """读 `config/wz.yaml`（叠加 `DEFAULTS` ✓）—— **按文件指纹缓存** ✓（2026-10-04 加 ✓）。

    为什么要缓存（用户 2026-10-04 ✓："点『添加』打开掉落物选择弹窗非常卡"）：
      本函数被**每一个** `xxx_path()` 调用（`drop_sprite_dir` / `drop_index_path` /
      `sprite_dir_path` … ✓），而它每次都**重新 open + `yaml.safe_load`** ✗（实测 **0.5 ms/次**
      ✓）⇒ 像"每个掉落物查一次图标路径"那种循环里，光这一层就能吃掉 0.5 ms/个 ✗
      （200 个格子 = 100 ms 纯读配置 ✓ 完全是白费 ✓）。
    ⚠ **失效判据 = `config/wz.yaml` 的 mtime/size 变了**（`_dir_stamp` ✓ 对文件同样适用 ✓）
      ⇒ 设置里改完配置**照样立刻生效** ✓（不许做成"改完要重启" ✗）。
    ⚠ 返回**拷贝**（同老行为：调用方随手改自己那份不影响别人 ✓）。
    """
    stamp = _dir_stamp(CONFIG_PATH)
    if _CFG_CACHE.get("stamp") == stamp and _CFG_CACHE.get("cfg") is not None:
        return dict(_CFG_CACHE["cfg"])
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(yaml.safe_load(f) or {})
        except Exception:
            pass
    _CFG_CACHE["stamp"] = stamp
    _CFG_CACHE["cfg"] = cfg
    return dict(cfg)


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


def map_name_of(map_id, path=None):
    """地图 id → **地图名**（`datasets/maps.json` ✓）—— 查不到 ⇒ `""` ✓（**绝不编** ✗）。

    ⭐ 用户 2026-10-04 ✓ 原话："A 机部署台的当前地图显示格式应该是 `{地图名}_{id}`" ✓
      —— 光给一串 `105040306` 人认不出是哪张图 ✓（而且现场那次"项目名字叫森林迷宫III、
      地图却是巨人之林"就是这么暴露出来的 ✓）。
    ⚠ 没有清单 / 查不到 ⇒ 回空串 ✓ 让调用方**只显示 id** ✓（编一个名字比不显示更坏 ✗）。
    ⚠ 带缓存（按文件指纹 ✓）：这文件 800 KB ✓，部署台那行是**每秒刷新**的 ✗。
    """
    p = maps_json_path() if path is None else Path(path)
    stamp = _dir_stamp(p)
    if _MAPNAME_CACHE.get("stamp") == stamp and _MAPNAME_CACHE.get("map") is not None:
        table = _MAPNAME_CACHE["map"]
    else:
        table = {}
        idx = load_index(str(p))
        for m in ((idx or {}).get("maps") or []):
            if isinstance(m, dict):
                mid = str(m.get("id") or "").strip()
                nm = str(m.get("name") or "").strip()
                if mid and nm:
                    table[mid] = nm
        _MAPNAME_CACHE["stamp"] = stamp
        _MAPNAME_CACHE["map"] = table
    return table.get(str(map_id or "").strip(), "")


def map_label(map_id, path=None):
    """地图的**显示名**：`{地图名}_{id}`（没有名字 ⇒ 就只是 id ✓）—— 口径**只此一处** ✓。

    见 `map_name_of` 里用户那句原话 ✓；部署台「当前地图」那行就用它 ✓。
    """
    mid = str(map_id or "").strip()
    if not mid:
        return ""
    nm = map_name_of(mid, path=path)
    return ("%s_%s" % (nm, mid)) if nm else mid


#: `map_name_of` 的缓存：`{"stamp": 文件指纹, "map": {id: 名}}`（见那个函数 ✓）。
_MAPNAME_CACHE = {}


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


def short_label(mid, name=""):
    """「当前地图」那种**一行确认**用的文字：`森林迷宫III_105040303`（用户 2026-10-02 给的样式 ✓）。

    ⚠ 与下拉用的 `format_label` **不是一回事**（别合并 ✗）：
      · `format_label` 是"**挑**的时候一眼认出"⇒ id 在前、还带怪名（可长 ✓）；
      · 本函数是"**确认我现在在哪张图**"⇒ **名字在前、id 在后、不吃怪名**（一行放得下 ✓）。
    名字查不到就只给 id ✓ —— 本套资源里有几百张图在 String.wz 里没名字，
    **别编一个**（"(未命名)_001000000" 比 "001000000" 更难认 ✗）。
    """
    n = str(name or "").strip()
    m = str(mid or "").strip()
    if n and m and n != m:
        return "%s_%s" % (n, m)
    return m or n


def map_entry(map_id, pool=None):
    """地图清单里那张图的条目（= `list_maps` 的一项）；空 id / 查不到 → `None` ✓。

    `pool` 可传现成的一份（调用方已列过就别再列一遍 ✓ 一处匹配口径 ✓）。
    """
    mid = str(map_id or "").strip()
    if not mid:
        return None
    rows = pool if pool is not None else list_maps(only_with_mob=False, keyword="")
    return next((x for x in rows if x["id"] == mid), None)


def apply_map_choice(project, map_id, pool=None):
    """把「选了这张图」写进项目 —— **唯一一处写口**（2026-10-02 收编 ✓）。

    谁在用：① 模型训练 →「识别目标」那张卡的下拉（`gui/steps/cards.MapCard` ✓）；
            ② 路线识别 →「寻路配置」顶部那个「手动更换」（用户 2026-10-02 ✓）。
    两处**必须走同一份**：这个动作不是"改一个字段"，而是四件一起写（少一件就是坑 ✓
    见下面），各写一份迟早分叉 ✗。

    写四件：`map_id` ✓ + 这张图的怪列表 `mobs` / `mob_names` ✓ + `mobs_cleared=False`
    （换了图，上一张"用户清空过怪"的记录作废 ✓），最后 `save()` ✓。

    返回那张图的清单条目（调用方可以拿它渲染一句人话 ✓）；
    **清单里没有（或没给项目）⇒ `None` 且一个字节都不写** ✗ ——
    半提交的后果很具体：项目里地图换了、怪列表还是上一张的 ⇒
    「确认要识别怪物」弹窗一个怪都没有（实测踩过）。
    """
    m = map_entry(map_id, pool=pool)
    if m is None or project is None:
        return None
    project.set("map_id", m["id"])
    project.set("mobs", list(m["mobs"]))
    project.set("mob_names", list(m["mob_names"]))
    project.set("mobs_cleared", False)
    project.save()
    return m


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


#: 取怪的动作帧时按这个顺序退：**很多飞的怪只有 fly**，没有 stand 就用 fly，
#: 连 move 都没有才落到「目录里任意一张 png」兜底。
MOB_ACTIONS = ("stand", "fly", "move")


def _frame_no(stem):
    """'stand_10' → 10。按帧号排，避免字符串排把 10 排到 2 前面。"""
    try:
        return int(stem.rsplit("_", 1)[1])
    except Exception:
        return 0


def mob_action_frames(mob_dir, actions=MOB_ACTIONS):
    """在怪目录里按优先级找一组动作帧 → (动作名, [路径…])；都没有 → (None, [])。

    **为什么需要它**：导出会把 img 里的**所有**动作都写出来，所以只有 fly 的怪
    在库里是 fly_0.png… 而没有 stand_*.png。以前各处写死 stand_*.png，
    这类怪在「确认要识别的怪物」里整类列不出来（缩略图也是空的），
    标定还会直接报「精灵库里找不到这些怪」。
    """
    if not mob_dir.is_dir():
        return None, []
    for a in actions:
        fs = sorted(mob_dir.glob("%s_*.png" % a), key=lambda p: _frame_no(p.stem))
        if fs:
            return a, fs
    fs = sorted(p for p in mob_dir.glob("*.png") if p.stem != "meta")
    return (None, fs) if fs else (None, [])


def list_sprite_mobs():
    """列出精灵库里所有**可用**的怪 id（排序，供手动选怪用）。

    判据是「有动作帧」，不是「有 stand」—— 很多**飞的怪只有 fly**
    （导出会把 img 里所有动作都写出来，所以它在库里是 fly_*.png）。
    以前写死 stand_*.png，这类怪在「确认要识别的怪物」里整类列不出来。
    """
    root = sprite_dir_path()
    out = []
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and mob_action_frames(d)[1]:
                out.append(d.name)
    return sorted(out)


# ══════════════════════════════════════════════════════════════
# 掉落物（道具）图库 —— 与「怪」那条线**对称**（用户 2026-10-04 ✓）
# ══════════════════════════════════════════════════════════════
# 为什么单开一条：怪那条走 `datasets/sprites/mob` + `maps.json`（名字散在各条地图记录里 ✓）；
#   而掉落物**没有"地图"这条线**（Map.wz 里没有"这张图会掉什么"✗）⇒ 名字只能来自**道具清单**
#   `datasets/drops.json`（`[{id, name}]` ✓），图来自 `datasets/sprites/drop/<id>/*.png` ✓。
# ⚠⚠ 这两样**现在都还不存在**：本仓库读 WZ 全靠外部 `WzProbe.exe`，而它现有子命令
#   （`tree` / `dump-mob` / `dump-terrain` / `dump-map` / `char-*`）**都写死了
#   Mob / Map / Character.wz** ⇒ 道具（`Item.wz`）**还没有导出通道** ✗（要等它加 `dump-item` ✓）。
#   ⇒ 所以下面一律**如实区分"没有"和"空"**：`drop_library_state()` 就是给界面那句实话 ✓
#     （只说"没搜到"会让人以为名字打错了 ✗）。
DEFAULT_DROP_SPRITE_DIR = "datasets/sprites/drop"
DEFAULT_DROP_INDEX = "datasets/drops.json"


def drop_sprite_dir(cfg=None):
    """掉落物图标目录（绝对路径；`config/wz.yaml` 的 `drop_sprite_dir` 可覆盖 ✓）。"""
    cfg = cfg or load_cfg()
    p = Path(cfg.get("drop_sprite_dir") or
             DEFAULTS.get("drop_sprite_dir", DEFAULT_DROP_SPRITE_DIR))
    return p if p.is_absolute() else (ROOT / p)


def drop_index_path(cfg=None):
    """掉落物清单 JSON 的路径（`[{id, name}]` ✓）。"""
    cfg = cfg or load_cfg()
    p = Path(cfg.get("drops_json") or
             DEFAULTS.get("drops_json", DEFAULT_DROP_INDEX))
    return p if p.is_absolute() else (ROOT / p)


def load_drop_index(path=None):
    """读掉落物清单 → `[{id, name}]`（文件没有/坏了/格式怪 ⇒ **空表** ✓ 不抛 ✗）。

    ⚠ 空表 = "没有这一项数据"，**不等于**"清单是空的"✗ —— 要区分两者请用
      `drop_library_state()`（它看的是**文件在不在** ✓）。
    接三种写法：`{"drops": [...]}` / 顶层数组 / `[[id, name], ...]` ✓（导出工具怎么方便怎么来 ✓）。
    """
    p = Path(path) if path else drop_index_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = raw.get("drops") or raw.get("items") or []
    out = []
    for it in (raw or []):
        if isinstance(it, dict) and str(it.get("id") or ""):
            out.append({"id": str(it["id"]), "name": str(it.get("name") or "")})
        elif isinstance(it, (list, tuple)) and it and str(it[0]):
            out.append({"id": str(it[0]),
                        "name": str(it[1]) if len(it) > 1 else ""})
    return out


def norm_drops(raw):
    """把界面/配置里的掉落物列表归一成 `[{"id", "name"}]` ✓ —— **一处口径**（卡片与弹窗共用 ✓）。

    ⚠ 得容忍三种历史写法：`["04030001", …]`（只存 id ✓）/ `[{"id":…, "name":…}]` ✓ /
      `[[id, name], …]` ✓ —— 少一种就会在某个环节悄悄丢项 ✗。
    """
    out = []
    for it in (raw or []):
        if isinstance(it, dict):
            did = str(it.get("id") or "").strip()
            nm = str(it.get("name") or "").strip()
        elif isinstance(it, (list, tuple)) and it:
            did, nm = str(it[0]).strip(), (str(it[1]).strip() if len(it) > 1 else "")
        else:
            did, nm = str(it or "").strip(), ""
        if did:
            out.append({"id": did, "name": nm})
    return out


def drop_name_map():
    """「掉落物 id → 名称」（清单里没名字的就不在里面 ✓）。"""
    return {d["id"]: d["name"] for d in load_drop_index() if d.get("name")}


def clean_desc(s):
    """把 WZ 的**描述文本**洗成人看的文字（**纯函数 ⇒ 可单测** ✓）。

    现场（2026-10-04 ✓ 全量导出 4549 条道具后统计）：3732 条有描述，里面混着三种**标记**✗——
      · 字面 `\\r\\n` / `\\n`（**两个字符**，不是换行 ✗）—— 21 + 804 条 ✓ ⇒ 换**真换行** ✓；
      · 真制表符（U+0009 ✓ 16 行）⇒ 换空格 ✓（也正是它把 JSON 写坏过 ✗ 见 `ItemDump.J` ✓）；
      · `#c…#`（Maple 的颜色码 ✓ 618 条）⇒ 去掉 ✓。
    ⚠ 其余**一律原样保留** ✗ —— 比如 `%d`/`%s` 这种占位（264 条 ✓）不去猜它的值 ✓
      （"50%" 这种正文本里也有 `%` ✓，乱替换会把正文改坏 ✗）。
    """
    t = str(s or "")
    if not t:
        return ""
    for a, b in (("\\r\\n", "\n"), ("\\n", "\n"), ("\\r", "\n")):
        t = t.replace(a, b)
    t = t.replace("\t", " ").replace("#c", "").replace("#", "")
    return t.strip()


def load_drop_meta(path=None):
    """清单里那几项**额外字段** → `{id: {"category", "frames", "icon_only"}}` ✓（弹窗拿它显示
    "类别 / 几帧 / 是不是退回用 icon 了" ✓）。

    ⚠ 与 `load_drop_index`（只归一 `{id, name}` ✓）**分开**：那个的返回值被用例按
      `{"id","name"}` 精确断言过 ⇒ 往它里面塞字段会把那次断言弄红 ✗。
    ⚠ 缺字段/读不到 ⇒ 空 dict ✓（界面只少一行说明，不该因此报错 ✗）。
    """
    p = Path(path) if path else drop_index_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict):
        raw = raw.get("drops") or raw.get("items") or []
    out = {}
    for it in (raw or []):
        if not isinstance(it, dict):
            continue
        did = str(it.get("id") or "")
        if not did:
            continue
        try:
            frames = int(it.get("frames") or 1)
        except (TypeError, ValueError):
            frames = 1
        out[did] = {"category": str(it.get("category") or ""),
                    "frames": max(1, frames),
                    "icon_only": bool(it.get("icon_only")),
                    # ⭐ **描述**（用户 2026-10-04 ✓ "把 desc 一并导出并做进显示功能"）——
                    #   出口处就洗好（`clean_desc` ✓）：消费方（信息窗 / 候选 / 弹窗）拿到的
                    #   一律是**能直接显示**的文字 ✓，别让每处各自 replace 一遍 ✗。
                    "desc": clean_desc(it.get("desc"))}
    return out


def drop_frame_files(drop_id):
    """某个掉落物的**全部**图标帧（没有 ⇒ `[]` ✓）。"""
    d = drop_sprite_dir() / str(drop_id)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.png") if p.stem != "meta")


def drop_icon_path(drop_id):
    """某个掉落物的**一张**图标（没有 ⇒ `None` ✓）—— 走**缓存的索引** ✓，不再每个 id 一次 glob ✗。

    ⚠ 语义与老实现**一字不差**（`drop_frame_files(id)` 排序后的第一张 ✓，`meta` 不算 ✓）——
      改的只是"怎么拿到它"✓（见 `drop_icon_index` 里那段实测数字 ✓）。
    """
    return drop_icon_index().get(str(drop_id))


#: 「4 种金币」的 id（图库没导出时的兜底 ✓ 见 `default_drop_ids` ✓）。
COIN_DROP_IDS = ("09000000", "09000001", "09000002", "09000003")
#: 掉落物图标索引的缓存：`{"key": (目录指纹 ✓), "map": {id: 第一帧路径}}`。
_DROP_ICON_INDEX = {}
#: `list_sprite_drops` 的"全量行"缓存：`{"key": (清单指纹, 目录指纹 ✓), "rows": [...]}`。
_DROP_ROWS_CACHE = {}


def _dir_stamp(p):
    """目录的"现在长这样"指纹（`mtime_ns` + `size` ✓）—— 缓存失效的**唯一**判据 ✓。

    ⚠ 拿不到（目录不在 / 没权限）⇒ `None` ✓ —— `None` 也是一个**合法指纹**（"那儿什么都没有"
      ✓），所以**别**把它当"缓存缺失" ✗（否则图库没导出的环境会每拍重建 ✓）。
    """
    try:
        st = Path(p).stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def clear_drop_caches():
    """把掉落物那几个缓存丢掉（**导出工具改完目录后可以自己调** ✓；等价于"下次现读" ✓）。

    ⚠ 一般情况下**不用调**：重导必然新建 id 目录 ⇒ `drop_sprite_dir()` 的指纹就变了 ✓
      （见 `_dir_stamp` ✓）。只有"**原地覆盖**同一个目录里的帧"（目录指纹不变 ✗）才需要它 ✓。
    """
    _DROP_ICON_INDEX.clear()
    _DROP_ROWS_CACHE.clear()


def drop_icon_index(root=None):
    """`{id: 第一帧 png 的绝对路径}` —— **一趟 scandir 建好 + 缓存** ✓（2026-10-04 加 ✓）。

    为什么必须有它（用户 2026-10-04 ✓ 原话："点『添加』打开掉落物选择弹窗非常卡"）：
      · 老实现 `drop_icon_path(id)` = **每个 id 一次 `glob("*.png")` + sorted** ✗，实测
        **1.3 ms/个** ✓ ⇒ 弹窗填 200 个格子 = **0.28 s** ✗、全量 4549 个理论上 **6 s** ✗；
      · 现在 = 建一次索引（每个目录**一趟** `scandir` ✓）⇒ 之后每次查是**字典取值** ✓
        （实测从 1.3 ms ⇒ 约 **0.002 ms** ✓）。
    ⚠ 语义照旧：**排序后的第一张**、`meta` 不算（= `drop_frame_files` 的老口径 ✓）；
      大小写和 Windows 的 `glob("*.png")` 一致（`.lower().endswith(".png")` ✓）。
    ⚠ 失效判据 = `drop_sprite_dir()` 的**指纹**变了（见 `_dir_stamp` ✓）。
    """
    r = Path(root) if root else drop_sprite_dir()
    key = (str(r), _dir_stamp(r))
    if _DROP_ICON_INDEX.get("key") == key and _DROP_ICON_INDEX.get("map") is not None:
        return _DROP_ICON_INDEX["map"]
    out, counts = {}, {}
    if r.is_dir():
        import os as _os

        for ent in _os.scandir(str(r)):
            try:
                if not ent.is_dir():
                    continue
                first, n = "", 0
                for f in _os.scandir(ent.path):
                    if not f.name.lower().endswith(".png"):
                        continue
                    if Path(f.name).stem == "meta":      # 同 `drop_frame_files` ✓
                        continue
                    n += 1                               # ⭐ 顺手数**磁盘上实际有几帧** ✓
                    if not first or f.name < first:      # = sorted(...)[0] ✓
                        first = f.name
                if first:
                    out[ent.name] = Path(ent.path) / first
                    counts[ent.name] = n
            except OSError:
                continue
    _DROP_ICON_INDEX["key"] = key
    _DROP_ICON_INDEX["map"] = out
    _DROP_ICON_INDEX["n"] = counts                  # 同一次扫描顺手收的 ✓ 不另扫 ✗
    return out


def drop_frame_counts():
    """`{id: 磁盘上实际有几帧}` ✓（与 `drop_icon_index` **同一次扫描**收的 ✓ 不另扫 ✗）。

    为什么要它（用户 2026-10-04 ✓ 现场："我移除了部分的 icon_3.png 以及 meta 数据（因为这个图
    太小了容易误标）"）：**清单里的 `frames` 是导出那一刻的数字** ✗ —— 用户手工删掉多余帧之后
    它还写着 4 ✓，信息窗照着念就成了假话 ✗。有了这一份，信息窗能如实说"**磁盘上 3 张**（清单写
    4 帧）"✓，人要核对"我删干净没有"**一眼就能看到** ✓（口径：标注只用**磁盘上现有的**这几张 ✓
    见 `drop_frame_files` ✓）。
    """
    drop_icon_index()
    return dict(_DROP_ICON_INDEX.get("n") or {})


def default_drop_ids():
    """**默认**要标注的掉落物 = 那 **4 种金币** ✓（用户 2026-10-04 ✓ 原话："将4种金币默认添加进
    要标注的掉落物"）。

    口径：图库里 **`09` 开头**的那些 id ✓（实测就是 `09000000` / `...01` / `...02` / `...03`
    这四个 ✓，每个 3~4 帧 ✓，清单里 `category` 都是 `Special` ✓）—— **从库里现取**，新增/改名
    自动跟上 ✓。
    ⚠ 图库**还没导出**（`datasets/sprites/drop` 不存在 / 没有 09 开头的）⇒ 退回
      `COIN_DROP_IDS` 那份**写死的四个 id** ✓：默认值本来就是"数据先行"的东西 ✓，
      不能因为图库没导就变成空 ✗（那等于没做这件事 ✓）。
    ⚠ 消费方（`gui/steps/cards.py`）**只在"这一项从来没存过"时**用它 ✓ —— 存过（哪怕被清成
      空表 ✓）就照用户存的来 ✓，否则"删掉金币"会被当成"没配过"、下次又长回来 ✗。
    """
    have = sorted(i for i in drop_icon_index() if str(i).startswith("09"))
    return have or list(COIN_DROP_IDS)


def drop_rows():
    """**全量**候选行（= `list_sprite_drops("")` 那一份 ✓）—— 一趟建好 + 缓存 ✓（2026-10-04 加 ✓）。

    为什么单独抽出来（用户 2026-10-04 ✓："点『添加』打开掉落物选择弹窗非常卡"）：
      · 弹窗打开要**全量**（空关键词浏览 ✓），打字时又要按词筛 ✓ —— 而老实现**每次调用都从头
        建 4549 行** ✗（实测 **0.29 s/次** ✓ 肉眼可感 ✓），弹窗一开就调两趟 ✗；
      · 现在建一次 ✓，筛只在内存里过一遍 ✓（返回**拷贝** ⇒ 调用方改不到缓存 ✗）。
    ⚠ 失效判据 = **清单文件 + 图标目录**的指纹（`_dir_stamp` ✓）—— 重导必然新建 id 目录 ✓。
    """
    idx_p, root = drop_index_path(), drop_sprite_dir()
    key = (str(idx_p), _dir_stamp(idx_p), str(root), _dir_stamp(root))
    if _DROP_ROWS_CACHE.get("key") == key and _DROP_ROWS_CACHE.get("rows") is not None:
        return _DROP_ROWS_CACHE["rows"]
    names = drop_name_map()
    meta = load_drop_meta()
    ids = set(names)
    have = drop_icon_index()            # ⚠ 用**同一份缓存索引**判"有没有图" ✓（不再另扫一趟 ✗）
    n_disk = drop_frame_counts()        # ⭐ 磁盘上实际几帧（同一次扫描 ✓ 见它的说明 ✓）
    ids |= set(have)
    out = []
    for i in sorted(ids):
        m = meta.get(i) or {}
        out.append({"id": i, "name": names.get(i, ""),
                    "has_img": i in have,
                    "category": str(m.get("category") or ""),
                    "frames": int(m.get("frames") or 1),
                    # ⭐ **磁盘上的真实帧数**（0 = 没有图 ✓）—— 信息窗拿它跟清单那份对比 ✓
                    #   （用户手工删过帧之后，清单会过期 ✗ 见 `drop_frame_counts` ✓）
                    "frames_disk": int(n_disk.get(i, 0)),
                    "desc": str(m.get("desc") or "")})   # ⭐ 信息窗要显示它（`icon_grid.tip_lines` ✓）
    _DROP_ROWS_CACHE["key"] = key
    _DROP_ROWS_CACHE["rows"] = out
    return out


def list_sprite_drops(keyword=""):
    """掉落物候选（**清单 ∪ 图标目录** ✓），按 `keyword` 筛 **id / 名称 / 类别 / 描述** ✓。

    返回 `[{id, name, has_img, category, frames, desc}]` ✓（后几项来自清单 ✓ 没有就是 `""` / 1 ✓
    —— 弹窗拿它们显示"类别 · 几帧"，"金币是 4 帧动画"这种事一眼能看到 ✓）。

    ⚠ 为什么要并集：清单给"WZ 里叫啥"（选得中 ✓）、图标目录给"有没有图"（标得了 ✓）——
      只认一边都会漏（有名字没图 ⇒ 标不了 ✗；有图没名字 ⇒ 搜不到 ✗）。
    ⚠ `keyword` 空 ⇒ **全列**（弹窗自己决定要不要列、列多少 ✓ 见
      `gui/drop_picker.py` 里那段"**不打字也能浏览**"的说明 ✓：金币在 String.wz 里
      **没有名字** ✗ ⇒ 逼人先打字＝逼人去猜 id ✓，所以这里必须支持"空关键词全列" ✓）。
    ⚠⭐ **全量行走 `drop_rows()` 的缓存**（2026-10-04 ✓ 见那个函数里的实测数字 ✓）——
      本函数只剩"内存里筛一遍 + 拷一份"✓，**不再每次重建 4549 行** ✗。返回的 dict 是**拷贝** ✓
      （调用方随手改自己的那份不会污染缓存 ✓）。
    """
    kw = str(keyword or "").strip().lower()
    rows = drop_rows()
    if not kw:
        return [dict(r) for r in rows]
    return [dict(r) for r in rows
            if (kw in r["id"].lower() or kw in r["name"].lower()
                or kw in r["category"].lower() or kw in r["desc"].lower())]


def show_drop(d):
    """列表里那一行的显示文本（有名字 ⇒「名字  (id)」✓ 同 `gui/mob_picker._mob_display` ✓）。"""
    nm = str((d or {}).get("name") or "").strip()
    i = str((d or {}).get("id") or "").strip()
    return ("%s  (%s)" % (nm, i)) if nm else i


def drop_library_state():
    """掉落物图库现状 → `(可用吗, 说明)` —— 给界面**一句实话** ✓（用户 2026-10-04 ✓）。

    ⚠ 为什么必须有它：图库现在**必然不存在**（外部 `WzProbe.exe` 还没有道具子命令 ✗）。
      界面若只显示"没有候选"，人第一反应是"我名字打错了" ✗ —— 所以要把"**哪一步缺**"
      直接写出来（清单路径 / 图标目录 / 还差哪个导出 ✓）。
    """
    idx_p, spr = drop_index_path(), drop_sprite_dir()
    n_idx = len(load_drop_index())
    n_img = len(list_sprite_drops()) if spr.is_dir() else 0
    if n_idx == 0 and not spr.is_dir():
        return False, (
            "掉落物图库还没导出：\n"
            "  清单 %s（没有）\n  图标目录 %s（没有）\n\n"
            "本仓库读 WZ 靠外部 WzProbe.exe，它现在**没有道具（Item.wz）子命令**\n"
            "（tree / dump-mob 都写死了 Mob.wz）⇒ 要先给它加一个 dump-item ✓。"
            % (idx_p, spr))
    if n_idx == 0:
        return False, ("有图标目录（%s）但**没有清单** %s ⇒ 名字/搜索用不了 ✓"
                       % (spr, idx_p))
    if n_img == 0:
        return False, ("有清单 %s（%d 项）但**图标目录是空的**（%s）⇒ 没图就标不了 ✓"
                       % (idx_p, n_idx, spr))
    return True, ("掉落物图库：清单 %d 项（%s）· 图标目录 %s"
                  % (n_idx, idx_p, spr))


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


# ══════════════════════════════════════════════════════════════
# 宠物 / 宠物装备图库的**补导**（用户 2026-10-05 ✓）
# ══════════════════════════════════════════════════════════════
# 为什么要这一块（用户 2026-10-05 ✓ 原话："如果没有导出，在添加的带装备的宠物、关闭弹窗后，
#   在数据集工作台显式读条导出"）：
#   组合外观（宠物 + 装备 ✓）是**按姿态分别叠**的 ✓，而装备图库**只导了 `stand0`** ✗
#   ⇒ 弹窗里除了 stand0 那些帧全是"纯本体"（现场 = 鳄鱼潭1 / 花蘑菇仔 ✓ 详见
#   `core/petlib.py` 里那段"组合要用的图导没导"✓）—— 源头是导出器 `dump-petequips` 的
#   `--states` **默认 `stand0`** ✓（`PetDump.cs::RunPetEquips` ✓）。
#   判据（缺哪几个姿态）在 `core.petlib.combo_export_plan` ✓ —— **这里只管把它跑出来** ✓
#   （一处口径 ✓ 界面层不自己拼命令 ✗）。

def pet_export_task(plan, cfg=None):
    """补导任务 ⇒ `(fn, params)`（与 `card.make_task` **同一个形状** ✓ 好让工作台直接跑 ✓）。

    `plan` 来自 `petlib.combo_export_plan` ✓（**空表 ⇒ 别调它** ✓ 调用方先判 ✓）。
    ⚠ `cfg` 只在这里读一次（exe / wz_dir ✓）⇒ 起任务时就把配置**冻进 params** ✓
      （别在子线程里再读一遍配置 ✗ —— 那样"跑到一半改了配置"会变成另一种行为 ✓）。
    """
    cfg = cfg or load_cfg()
    return run_pet_export, {
        # ⚠ **连里面的列表也拷一份**（浅拷 `dict(s)` 会把 `ids`/`states` 共享出去 ⇒
        #   调用方回头改自己那份，就等于改**另起线程里正跑着的那份** ✗）。
        "plan": [{"cmd": s.get("cmd"), "ids": list(s.get("ids") or []),
                  "states": list(s.get("states") or [])} for s in (plan or [])],
        "exe": find_probe_exe(cfg.get("probe_exe") or ""),
        "wz_dir": cfg.get("wz_dir") or "",
    }


def pet_export_cmd(exe, step, wz_dir, out):
    """一条 `dump-pets` / `dump-petequips` 命令（**纯函数** ⇒ 用例可以逐字钉 argv ✓）。"""
    cmd = [str(exe), str(step.get("cmd") or ""), str(wz_dir), str(out),
           "--only", ",".join(str(x) for x in (step.get("ids") or []))]
    sts = [str(s) for s in (step.get("states") or [])]
    if sts:                                   # ⚠ 本体那条**没有** `--states`（一趟全导 ✓）
        cmd += ["--states", ",".join(sts)]
    return cmd


def run_pet_export(params, ctx=None):
    """按计划补导宠物图库（逐条跑 ✓ 一条失败就停并如实报 ✓）；返回 `{"summary": …}` ✓。

    params: plan（`petlib.combo_export_plan` 的产物 ✓）/ exe / wz_dir
    ⚠ 进度是**忙式**的（`progress(0, 0, …)` ✓）：导出器装备那条**只打印收尾一行** ✗
      ⇒ 没有真实百分比 ⇒ **不编** ✗（同 `gui/live_panel` 里 TensorRT 那处的纪律 ✓）。
    ⚠ 导完**不用手删** `datasets/sprites/pet_combo/` ✓：`compose_pet` 的缓存指纹看的是
      **源目录的 mtime/size** ✓（`petlib._stamp` ✓）⇒ 源一变，下一趟自动重做 ✓。
    """
    from core import petlib            # ⚠ 循环导入：`petlib` 反过来 import 本模块 ⇒ 只能在这儿 ✓
    ctx = ctx or TaskContext()

    plan = [s for s in (params.get("plan") or []) if (s.get("ids"))]
    if not plan:
        return {"summary": "没有要补导的东西"}

    cfg = load_cfg()
    exe = find_probe_exe(params.get("exe") or cfg.get("probe_exe") or "")
    wz_dir = params.get("wz_dir") or cfg.get("wz_dir") or ""

    if not exe:
        raise FileNotFoundError(
            "找不到 WzProbe.exe（补导宠物图库要靠它 ✓）。\n"
            "请先编译:\n"
            "  cd E:\\MyPrograms\\RippleRogue\\MapleNecrocer\n"
            "  dotnet build WzProbe/WzProbe.csproj -c Release\n"
            "然后在「WZ 导出」对话框里指定它的路径。")
    if not Path(exe).exists():
        raise FileNotFoundError("WzProbe.exe 不存在: %s" % exe)
    # 空串必须单独判（同 `run_export_maps` ✓：`Path("")` 在 Windows 上等于 `Path(".")` ✓）
    if not wz_dir:
        raise ValueError("没有指定 WZ 目录（在「WZ 导出」对话框里设置，或写进 config/wz.yaml）")
    if not Path(wz_dir).is_dir():
        raise FileNotFoundError("WZ 目录不存在: %s" % wz_dir)

    outs = {"dump-pets": petlib.pet_dir(), "dump-petequips": petlib.pet_equip_dir()}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    done = []

    for step in plan:
        name = str(step.get("cmd") or "")
        out = outs.get(name)
        if out is None:
            raise ValueError("不认识的导出子命令: %r（只认 %s）"
                             % (name, " / ".join(sorted(outs))))
        Path(out).mkdir(parents=True, exist_ok=True)
        cmd = pet_export_cmd(exe, step, wz_dir, out)
        what = "本体" if name == "dump-pets" else "装备"
        sts = ",".join(str(s) for s in (step.get("states") or []))
        ctx.log("补导宠物%s：%s  —— %s" % (what, "、".join(step["ids"]),
                                          ("姿态 " + sts) if sts else "全部姿态"))
        ctx.log("  跑: " + " ".join(cmd))
        ctx.log("  输出: %s" % out)
        ctx.progress(0, 0, "导出宠物" + what)      # 忙式（导出器不给百分比 ⇒ 不编 ✓）

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
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait()

        if canceled:
            ctx.log("已取消补导", "warn")
            return {"summary": "已取消"}
        if proc.returncode != 0:
            raise RuntimeError("WzProbe 退出码 %d（%s 失败）" % (proc.returncode, name))
        done += [str(x) for x in step["ids"]]

    ctx.log("── 补导完成 ──", "ok")
    ctx.log("  %d 项：%s" % (len(done), "、".join(done)))
    ctx.log("  组合模板下一趟会自动重做 ✓（源目录指纹变了 ✓）")
    ctx.progress(1, 1, "完成")
    return {"summary": "补导 %d 项（%s）" % (len(done), "、".join(done)), "ids": done}


def run_export_terrain(params, ctx=None):
    """导出**地形**（寻路用）：`datasets/map/<id>.json` + 小地图底图 `<id>.png`。

    params: map_id（必须）/ exe / wz_dir / out（默认 datasets/map）

    **只导这一张**（`--only <id>`）：不带 id 会把 Map.wz 里三千多张全导一遍
    （几十秒到几分钟）—— 从界面上点一下等那么久不合适；单张通常几秒。
    写法与 run_export_maps 一致：起子进程、stdout 逐行转日志、可取消。
    """
    ctx = ctx or TaskContext()

    mid = str(params.get("map_id") or "").strip()
    if not mid:
        raise ValueError("没有指定地图 id")

    cfg = load_cfg()
    exe = find_probe_exe(params.get("exe") or cfg.get("probe_exe") or "")
    wz_dir = params.get("wz_dir") or cfg.get("wz_dir") or ""
    out = params.get("out") or str(ROOT / "datasets" / "map")

    if not exe:
        raise FileNotFoundError(
            "找不到 WzProbe.exe —— 在 config/wz.yaml 里填 probe_exe，"
            "或把 exe 放回默认位置")
    if not wz_dir or not Path(wz_dir).is_dir():
        raise FileNotFoundError(
            "WZ 目录不存在: %s（在 config/wz.yaml 里配 wz_dir）" % (wz_dir or "（空）"))

    Path(out).mkdir(parents=True, exist_ok=True)

    ctx.log("WzProbe  : %s" % exe)
    ctx.log("WZ 目录  : %s" % wz_dir)
    ctx.log("输出     : %s" % out)
    ctx.log("导出地形 %s（只导这一张）…" % mid)

    cmd = [exe, "dump-terrain", wz_dir, str(out), "--only", mid]

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
        raise RuntimeError("WzProbe 退出码 %d，地形导出失败" % proc.returncode)

    j = Path(out) / ("%s.json" % mid)
    png = Path(out) / ("%s.png" % mid)
    if not j.exists():
        raise RuntimeError(
            "导出结束了，但找不到 %s —— 这张 id 可能不在 Map.wz 里" % j)

    has_png = png.exists()
    ctx.log("── 地形导出完成 ──", "ok")
    ctx.log("  %s" % j.name)
    ctx.log("  %s" % (("%s（小地图底图）" % png.name) if has_png else
                      "（这张图没有 miniMap 节点 → 只有地形，没有底图）"))
    return {"summary": "已导出 %s" % mid,
            "json": str(j), "png": str(png) if has_png else ""}
