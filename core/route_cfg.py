"""「路线识别页签」那批配置按**地图 id** 的持久化（2026-10-03 ✓ 用户要求）。

背景（用户原话 ✓）：
  · 2026-10-02："**路线识别页签所有配置按地图 id 存**"；
  · 2026-10-03 逐项审计后确认口径：这批键改成按图，**`project.yaml` 里那几格留下**
    当"**新图第一次打开时的播种值**"✓（这样新开一张图不至于全是硬编码默认 ✓）；
  · 「地形图」里叠加层的**透明度保持现状**（全局本机 `config/live.yaml` ✓ 纯显示偏好 ✓）。

落点：`datasets/map/<id>.route.json`（和 `zones.json` / `.battle.json` / `.mapcalib.json`
并列 ✓）。分两拨：
  · `PARAM_KEYS` —— **`DecisionSettings` 上的参数**（攀爬参数 / 移动重试 / 起跳距离）；
  · `MAP_KEYS`   —— **面板自己持有的那三样**（小地图框选区域 / 来源 / 黄点跟踪参数）
    ⚠ 它们原来是 `project.yaml` 的 `mmap_crop` 与 `live.yaml` 的 `mmap_src`/`mmap_track`
      （**按项目 / 全局** ✗）⇒ 播种时从**那两处老家**读一次 ✓（迁移不丢配置 ✓）。

⚠ 分层（同 `core/battle.py` ✓）：core（第 0 层）不许 import decision/gui ⇒ 本模块
  **只做「读文件 / 写文件 / 路径 / 键清单」**，不碰钳位清洗（那是 `DecisionSettings.
  apply_route_cfg` 的活 ✓）、也不认识面板控件 ✗。
"""

import json

from core import mapdata

#: `DecisionSettings` 上的参数键（灌值由 `DecisionSettings.apply_route_cfg` 一处做 ✓）
PARAM_KEYS = (
    "climb_align_gap_ms",       # 对齐绳梯移动延迟(ms)
    "climb_align_near_px",      # 开始对齐绳梯x的距离(px)
    "climb_retry_delay_s",      # 攀爬失败：重试延时(s)
    "climb_retry_delay_inc_s",  # 攀爬失败：每次重试递增(s)
    "move_retry_ms",            # 移动操作尝试间隔(ms)
    "jump_start_px",            # 起跳距离(px)
    # ⭐ 「**禁用杀怪寻路**」（2026-10-06 用户定口径 ✓ 原话："**从此 禁用杀怪寻路就是按地图id存
    #   的数据，而不是在设置里全局一份**"）—— 它原来跟**项目**走（`project.yaml` 的 `decision`
    #   段 ✗）⇒ 现在按图 ✓（登记在这儿就自动三处齐 ✓：读 ✓ 写 ✓ 播种 ✓）。
    #   ⚠ 为什么该按图：它决定的是"**这张图**要不要用寻路追怪"（寻路成不成立是**图**的属性 ✓
    #     —— 同一张图上两根绳/几层平台的事 ✗），跟"这个项目"不是一回事 ✓。
    #   ⚠ `project.yaml` 里那一格**留着**当**播种值**（同上面 6 个 ✓）：老项目已经配过的那份
    #     会在某张图**第一次打开**时被种进 `<id>.route.json` ✓（迁移不丢 ✓）。
    "disable_chase_pathfinding",  # 禁用杀怪寻路（按图 ✓）
)

#: 面板自己持有的三样（不是 settings 属性 ✓ 由 `RoutePanel` 读写 ✓）
MAP_KEYS = (
    "mmap_crop",                # 小地图框选区域（B 机画面上那块面板的位置 ✓）
    "mmap_src",                 # 小地图来源：stream（A 机推流）/ live（实时画面裁）
    "mmap_track",               # 黄点跟踪参数（搜索半径 / 跳变上限… 那排 ✓）
)

#: 这份文件能存下来的全部键（加新参数时**记得登记** —— 用例会扫面板上的键来卡漏 ✓）
KEYS = PARAM_KEYS + MAP_KEYS


def path(map_id):
    """这张图的「路线识别页签配置」文件：`datasets/map/<id>.route.json`。"""
    return mapdata.map_dir() / ("%s.route.json" % map_id)


def load(map_id):
    """读某张图那份 → `dict` | `None`。

    ⚠ `None` = **文件不存在**（这张图还没配过 ⇒ 调用方要**播种** ✓）；
      `{}` = 文件在但里面一个键都没有（明确"空"）⇒ **不播种** ✓ —— 别把"没配过"和
      "配过又清空"混成一件事 ✗（同 `core.battle.load` 的口径 ✓）。
      认不出来的键**原样带回来**（老版本多写的键别丢 ✓ 以后要迁移也用得上 ✓）。
    """
    p = path(map_id)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                        # noqa: BLE001 —— 坏了当"没这份"，别崩 ✗
        return None
    if isinstance(raw, dict):
        got = raw.get("values")
        if isinstance(got, dict):
            return dict(got)
        # 老写法 / 手写文件：整个顶层就当值 ✓（排除 `map_id` 这种元信息键 ✓）
        return {k: v for k, v in raw.items() if k != "map_id"}
    return None


def save(map_id, values, base=None, conflicts=None):
    """把（调用方已收拾好的）值写进 per-map 文件。

    ⚠ 写**全部**给进来的键（包括"认不出来的"✓）：这样"手上那份"是完整的，
      下次读回来不会因为漏写某项而退回默认 ✗。

    ⭐⭐ **多实例（用户 2026-10-09 原话："我们必须要支持能开 2 个工作台"）**：
    给了 `base`（= 调用方**上次从这份文件里见到的那份** ✓ 见 `RoutePanel._load_route_cfg`
    记的 `_route_cfg_base` ✓）就做**读-改-写** —— 只把"**相对 `base` 改过的键**"盖回磁盘 ✓，
    另一个窗口刚改的键**留住** ✓。
      · 两边改了**同一个**键 ⇒ 后写的赢 ✓（无法两全 ✓）＋ 名字记进 `conflicts`（list ✓
        由调用方决定要不要说给用户听 ✓）；
      · `base=None` ⇒ **老行为**（整份写 ✓ 新建 / 没记基准时用它 ✓）。
    为什么这个文件也要管：`main_window` 的保存钩子在**任何** `settings.save()` 之后都会顺手
    调 `_save_route_cfg()` ✓ ⇒ 另一个窗口只是动了别的参数，也会把这 9 个键整份盖一遍 ✗。
    """
    vals = {str(k): v for k, v in (values or {}).items()}
    if base is not None:
        cur = load(map_id)                      # 磁盘上现在那份（另一个窗口可能刚写过 ✓）
        if isinstance(cur, dict):
            kept = {str(k): v for k, v in cur.items()}
            for k, v in vals.items():
                if k in base and base.get(k) == v:
                    continue                    # 我没改过它 ⇒ 磁盘上那份说话 ✓
                if (k in kept and k in base and base.get(k) != kept[k]
                        and kept[k] != v and conflicts is not None):
                    conflicts.append("route_cfg:" + k)   # 两边都改过 ⇒ 记账 ✓
                kept[k] = v
            vals = kept
    p = path(map_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"map_id": str(map_id), "values": vals},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    return p
