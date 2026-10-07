"""宠物 / 宠物装备图库 + **组合外观模板**（用户 2026-10-04 ✓ 选的方案 C）。

## 这套东西是干什么的

标注类别表里早就有 `CLASS_PET = 5`（`perception/classes.py` ✓"宠物 = 干扰类，标它只为让模型
学会区分**宠物 vs 怪**"✓）。要标它就得有**模板** —— 而宠物的样子 = **本体 + 戴的装备** ✓
（MapleNecrocer 的 `PetForm` 就是"点一只宠物 ⇒ 右边列它可戴的装备 ⇒ 点装备看合体效果"✓）。
所以用户要的落点是：**"要标注的宠物列表"里存的是「宠物 + 装备」的组合外观** ✓（方案 C ✓）。

## 数据从哪来（外部 `WzProbe` 导出，与怪物/掉落同一套格式 ✓）

    datasets/pets.json                  [{"id","name"}]                 （宠物名，String.wz/Pet.img ✓）
    datasets/pets_name_fixes.json       {"id": "正确名"}                 （⭐ 手改表：客户端那份
                                                                          String.wz 名字重了/错了 ⇒
                                                                          在这里按 id 纠正 ✓
                                                                          见 `load_pet_name_fixes` ✓）
    datasets/pet_equips.json            [{"id","name","pets":[宠物id…]}] （装备名 String/Eqp.img/Eqp/PetEquip/<id>/name ✓
                                                                          "能戴的宠物" = Character/PetEquip/<装备>/<宠物id> 子目录 ✓）
    datasets/sprites/pet/<id>/          icon.png（列表用图标 ✓）+ stand_0.png… + **meta.tsv**（origin/delay ✓）
    datasets/sprites/pet_equip/<id>/    同上 ✓

⚠ `meta.tsv` 的列与**怪物精灵库一模一样**（`#file originX originY ltX ltY rbX rbY delay` ✓
  见 `datasets/sprites/mob/<id>/meta.tsv` ✓）—— 这不是巧合：**组合图也照这个格式产出** ⇒
  下游（模板匹配那套）**一个字都不用改** ✓✓。

## 组合规则（照 MapleNecrocer 的 `Client/Pet.cs` ✓ 别自己发明 ✗）

    `class PetEquip : Pet` ⇒ **装备与宠物同锚点**，各自按自己那一帧的 `origin` 摆放 ✓
    （`Pet.cs`：`origin = ImageNode.GetNode("origin")` ⇒ `Offset = -origin` ✓）
  ⇒ **组合 = 两张（或多张）图按 origin 对齐叠起来** ✓，宠物在下、装备在上 ✓（绘制顺序 ✓）。
  具体到坐标：某张图在以 origin 为原点的那套坐标里占
      x ∈ [-ox, -ox + w]、y ∈ [-oy, -oy + h] ✓
  ⇒ 组合画布 = 所有图那堆矩形的并集 ✓、**组合图的 origin = (-min_x, -min_y)** ✓（照老口径 ✓）。

⚠ 少一帧怎么办：装备的帧数常比宠物少 ✓ ⇒ **按 `i % 装备帧数` 取**（拿最近的一帧顶上 ✓，
  绝不因为"缺第 3 帧"就整只不标 ✗）；宠物那一帧没有 ⇒ 这一帧直接跳过 ✓。
"""
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from core import wzexport

#: 图库相对仓库根的路径（`config/wz.yaml` 里同名键可覆盖 ✓ 与掉落物那套同一套写法 ✓）
DEFAULT_PET_DIR = "datasets/sprites/pet"
DEFAULT_PET_EQUIP_DIR = "datasets/sprites/pet_equip"
DEFAULT_PET_INDEX = "datasets/pets.json"
DEFAULT_PET_EQUIP_INDEX = "datasets/pet_equips.json"
#: 宠物名的**手改表**（客户端 `String.wz` 里名字重了/错了 ⇒ 在这里按 id 纠正 ✓
#:  ⚠ 必须放**本仓库**：重导 `dump-pets` 会覆盖 `pets.json` ✗ 见 `load_pet_name_fixes` ✓）。
DEFAULT_PET_NAME_FIXES = "datasets/pets_name_fixes.json"

#: 默认做组合用的动作。⭐ **`stand0`** = WZ 里那个状态名本身 ✓
#: （`Item.wz/Pet/<id>.img/stand0/<n>` ✓ 装备那边 `Character.wz/PetEquip/<装备>/<宠物id>/stand0/<n>` ✓
#:   —— 两边**同名配对** ✓ 见 `WzProbe/PetDump.cs` 的 `BodyStates` ✓；导出也**只导 stand0** ✓）
DEFAULT_STATE = "stand0"

#: 组合外观**默认做哪几个姿态**（用户 2026-10-04 ✓ 原话："多姿态进组合外观" ✓）。
#:
#: ⚠⚠ **为什么必须多姿态**（同一批 12 帧、同一套参数实测 ✓，见 SKILL 211 ✓）：
#:   · 只做 `stand0`（老行为 ✗）⇒ **1 框 / 有检出 8%** ✗ —— 宠物一走路就一个都匹配不上 ✓；
#:   · 加上 `move` 4 帧 ⇒ **8 框 / 42%** ✓✓（逐框裁图核对过，全是真的 ✓）。
#:   这正是 `tools/detect_mobs._pick_frames` 里说的那种"**看不见的洞**" ✓：宠物在画面上
#:   大部分时间在走 ✓，只拿站立帧当模板 ⇒ 那些帧永远标不到、训练集里一个样本都没有 ✓。
#:   `stand1` = 站立的第二套（呼吸动画 ✓ 也常见 ✓，一起做 ✓）。
#: ⭐ **2026-10-04 第二次扩**（用户原话："宠物的匹配站立 stand、走路 move、跳 jump、饿 hungry
#:   都需要是匹配的对象"✓）⇒ 在 `stand0`/`stand1`/`move` 之上**再加 `jump` / `hungry`** ✓：
#:   这两个姿态在真画面里同样常见 ✓（跳跃 1~2 帧、`hungry` 帧多 ✓）⇒ 都该进模板 ✓。
#: ⚠ **不是越多越好** ✗：模板数 = 每帧匹配次数（耗时 ✓）+ 假匹配的机会 ✓
#:   ⇒ 想再扩（`rest0`/`sit`/`angry`）就在这里加 ✓，但要盯着检出率与耗时 ✓
#:   （⚠ 执行端 `detect_mobs.load_templates` 拿的是 `sorted(d.glob("*.png"))` ✓ 而且**受
#:     `per_mob` 上限**（任务参数 ✓ 默认 20 ✓）⇒ 姿态一多，`_pick_frames` 会**抽样** ✗
#:     ⇒ 真要用满这些帧，得把「每只宠物的模板上限」也抬上去 ✓）。
COMBO_STATES = ("stand0", "stand1", "move", "jump", "hungry")


def _cfg_path(key, default):
    cfg = wzexport.load_cfg()
    p = Path(cfg.get(key) or wzexport.DEFAULTS.get(key, default))
    return p if p.is_absolute() else (wzexport.ROOT / p)


def pet_dir():
    """宠物本体精灵目录（`datasets/sprites/pet` ✓）。"""
    return _cfg_path("pet_sprite_dir", DEFAULT_PET_DIR)


def pet_equip_dir():
    """宠物装备精灵目录（`datasets/sprites/pet_equip` ✓）。"""
    return _cfg_path("pet_equip_sprite_dir", DEFAULT_PET_EQUIP_DIR)


def pet_index_path():
    """宠物清单 json（`[{"id","name"}]` ✓）。"""
    return _cfg_path("pets_json", DEFAULT_PET_INDEX)


def pet_equip_index_path():
    """宠物装备清单 json（`[{"id","name","pets":[…]}]` ✓）。"""
    return _cfg_path("pet_equips_json", DEFAULT_PET_EQUIP_INDEX)


def pet_name_fixes_path():
    """宠物名的**手改表** json（`{id: 正确名, "_note": 说明}` ✓）—— 见 `load_pet_name_fixes` ✓。"""
    return _cfg_path("pet_name_fixes", DEFAULT_PET_NAME_FIXES)


def _load_rows(path, with_pets=False):
    """读清单（**带缓存** ✓ 指纹 = 文件 mtime/size ✓ 与掉落物那套同一把尺子 ✓）。

    ⚠ 坏文件/没有 ⇒ **空表** ✓（界面那边由 `pet_library_state()` 说清"缺哪一步" ✓、
      绝不在这里抛 ✗）。
    """
    path = Path(path)
    stamp = wzexport._dir_stamp(path)
    key = (str(path), stamp, bool(with_pets))
    hit = _INDEX_CACHE.get("key")
    if hit == key and _INDEX_CACHE.get("rows") is not None:
        return _INDEX_CACHE["rows"]
    rows = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001 —— 读不到就当没有 ✓
        raw = None
    if isinstance(raw, dict):
        raw = raw.get("pets") or raw.get("items") or raw.get("equips") or []
    for it in (raw or []):
        if not isinstance(it, dict):
            continue
        did = str(it.get("id") or "").strip()
        if not did:
            continue
        row = {"id": did, "name": str(it.get("name") or "").strip()}
        if with_pets:
            row["pets"] = [str(x).strip() for x in (it.get("pets") or []) if str(x).strip()]
        rows.append(row)
    _INDEX_CACHE["key"] = key
    _INDEX_CACHE["rows"] = rows
    return rows


def load_pet_name_fixes():
    """**宠物名的手改表** → `{id: 正确名}` ✓（读不到 / 坏了 ⇒ 空 ✓ 不抛 ✗；带缓存 ✓）。

    ⭐ 为什么需要它（用户 2026-10-04 ✓ 原话："宠物5000042、5000046名称错了，正确的是「花蘑菇仔」，
    现在是雪娃娃"）：**客户端自己的 `String.wz/Pet.img` 里名字就是错的/重的** ✗ ——
    实测拿真 WZ 读一遍 ✓：`Pet.img` 里 5000041/42/43/46 **四个 id 全叫「雪娃娃」** ✗，
    可把贴图并排画出来一看 ✓：5000041 = 雪人 ✓、**5000042 = 橙色蘑菇** ✓、5000043 = 刺猬 ✗、
    5000046 = 橙色蘑菇 ✓（42 与 46 的像素指纹**一模一样** ⇒ 同一只模型 ✓）。
    ⇒ 这不是我们导错 ✓（导出忠实照抄 WZ ✓），是**源数据就错** ✗ ⇒ 只能在**我们这侧**纠正 ✓，
      而且必须放**本仓库**（`datasets/pets_name_fixes.json` ✓）——重导（`dump-pets`）会覆盖
      `pets.json` ✗，放那边就白改了 ✓。
    ⚠ 只在 `load_pet_index()` 这个**唯一漏斗**上生效 ✓（界面、卡片、信息窗都吃它 ✓）。
    """
    path = Path(pet_name_fixes_path())
    stamp = wzexport._dir_stamp(path)
    if _FIX_CACHE.get("stamp") == stamp and _FIX_CACHE.get("map") is not None:
        return _FIX_CACHE["map"]
    out = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001 —— 没有/坏了都当没这回事 ✓
        raw = None
    if isinstance(raw, dict):
        for k, v in raw.items():
            k = str(k).strip()
            # ⚠ 跳过 `_note` 之类的说明键 ✓（它们不是 id ✓ 但绝不进映射表 ✓）
            if k and not k.startswith("_") and str(v).strip():
                out[k] = str(v).strip()
    _FIX_CACHE["stamp"] = stamp
    _FIX_CACHE["map"] = out
    return out


def load_pet_index():
    """宠物清单 → `[{id, name}]` ✓（带缓存 ✓）。

    ⭐ 名字**过一遍手改表**（`load_pet_name_fixes` ✓）—— 客户端 `String.wz` 里那些错的/重的
      名字（如 5000042/5000046 明明是花蘑菇仔却叫雪娃娃 ✗）在这里被纠正 ✓；
      没被改过的 id 一字不动 ✓（**不改** `datasets/pets.json` 那份导出 ✓ 导出保持"原样 ✓"）。
    """
    fixes = load_pet_name_fixes()
    out = []
    for r in _load_rows(pet_index_path()):
        d = dict(r)
        nm = fixes.get(str(d.get("id") or ""))
        if nm:
            d["name"] = nm
        out.append(d)
    return out


def load_pet_equip_index():
    """宠物装备清单 → `[{id, name, pets}]` ✓（`pets` = 能戴它的宠物 id ✓ 见模块说明 ✓）。"""
    return [dict(r) for r in _load_rows(pet_equip_index_path(), with_pets=True)]


def pet_library_state():
    """宠物图库现状 → `(可用吗, 说明)` —— 给界面**一句实话** ✓（照 `drop_library_state` ✓）。

    ⚠ 为什么必须有它：图库现在**还没导出**（`WzProbe` 那边要加宠物/装备两个通道 ✓）
      ⇒ 界面若只说"没有候选"，人会以为名字打错了 ✗ ⇒ 必须把"**缺哪一步**"写出来 ✓。
    """
    idx, spr = pet_index_path(), pet_dir()
    eq_idx, eq_spr = pet_equip_index_path(), pet_equip_dir()
    n_idx, n_eq = len(load_pet_index()), len(load_pet_equip_index())
    if n_idx == 0 and not spr.is_dir():
        return False, (
            "宠物图库还没导出：\n"
            "  清单 %s（没有）\n  精灵目录 %s（没有）\n\n"
            "要 `WzProbe` 先加两个通道（与怪物/掉落同一套导出 ✓）：\n"
            "  · `dump-pets`      —— Item.wz/Pet/<id>（图标 + stand/move 帧 + origin ✓）\n"
            "  · `dump-petequips` —— Character.wz/PetEquip/<装备>（图标 + 帧 + origin ✓，\n"
            "                        并记下它下面挂着哪些宠物 id ⇒ 右栏「能戴的装备」靠它 ✓）\n"
            "⚠ 现在连**名字**也读不到（String.wz/Pet.img 那 51 条要一起导出 ✓）。"
            % (idx, spr))
    if n_idx == 0:
        return False, ("有宠物精灵目录（%s）但**没有清单** %s ⇒ 名字/搜索用不了 ✓"
                       % (spr, idx))
    if n_eq == 0:
        return False, ("宠物本体齐了（%s 共 %d 项）但**宠物装备清单是空的** %s ⇒ "
                       "右栏会是空的、也做不出组合外观 ✓" % (idx, n_idx, eq_idx))
    return True, ("宠物图库：宠物 %d 项（%s）· 装备 %d 项（%s）· 精灵目录 %s / %s"
                  % (n_idx, idx, n_eq, eq_idx, spr, eq_spr))


#: 清单缓存（`{"key":…, "rows":…}` ✓）。⚠ 与掉落物那边**分开**：键里带 `with_pets` ✓ 别串味 ✗
_INDEX_CACHE = {}
#: 组合模板的缓存键（`{(宠物, 装备们, 动作): 源指纹}` ✓）—— 源变了就重做 ✓
_COMBO_STAMP = {}
#: 宠物名**手改表**的缓存（`{"stamp": 文件指纹, "map": {id: 名}}` ✓ 见 `load_pet_name_fixes` ✓）
_FIX_CACHE = {}


def clear_pet_caches():
    """清宠物这边的缓存（清单 + 组合指纹 ✓）—— 导完新数据可以调 ✓（一般不用：看指纹 ✓）。"""
    _INDEX_CACHE.clear()
    _COMBO_STAMP.clear()
    _FIX_CACHE.clear()          # ⭐ 手改表也一起清（用例换临时图库时不留上一份 ✓）


# ══════════════════════════════════════════════════════════════
# 帧 + origin（读 `meta.tsv` ✓ 与怪物精灵库同一个格式 ✓）
# ══════════════════════════════════════════════════════════════
def _parse_meta(mp):
    """读 `meta.tsv` → `{文件名: (originX, originY, delay)}` ✓（缺/坏 ⇒ 空 ✓ 不抛 ✗）。

    ⚠ 列顺序是**定的**（怪物那边导出就是它 ✓）：`#file originX originY ltX ltY rbX rbY delay`
      ⇒ 只取第 1/2/3 列 + 最后一列 ✓，中间那几列（包围盒）用不上 ✓。
    """
    out = {}
    try:
        lines = Path(mp).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            ox, oy = float(parts[1]), float(parts[2])
        except (TypeError, ValueError):
            continue
        try:
            delay = float(parts[-1])
        except (TypeError, ValueError):
            delay = 0.0
        out[parts[0]] = (ox, oy, delay)
    return out


def sprite_frames(root, sid, state=DEFAULT_STATE, pet_id=None):
    """某个 id 的**某一动作**所有帧 → `[(Path, ox, oy)]`（按帧号排序 ✓ 没有 ⇒ `[]` ✓）。

    ⚠ 只认 `<state>_<n>.png` 这种命名（与怪物/掉落同一套 ✓）；`icon.png` 不算帧 ✓。
    ⭐⭐ `pet_id`（**宠物装备专用** ✓）：装备的美术**按宠物分开存** ✓ ——
      `Character.wz/PetEquip/<装备>/<宠物id>/stand0/<n>` ✓（见 `WzProbe/PetDump.cs` ✓），
      导成 `<根>/<装备id>/<宠物id>/stand0_<n>.png` ✓。
      · 传了 `pet_id` ⇒ 只看 `<根>/<装备id>/<宠物id>/` ✓；**没有那个子目录 = 这只宠物戴不了它** ✓
        （回空表 ⇒ 组合时跳过 ✓ —— 比"硬套别的宠物那套图"正确得多 ✓）；
      · 不传（宠物本体 ✓）⇒ 还是 `<根>/<id>/` ✓（老行为 ✓）。
    """
    d = Path(root) / str(sid)
    if pet_id is not None:
        d = d / str(pet_id)
    if not d.is_dir():
        return []
    # ⚠⚠ `meta.tsv` 在**哪一层**取决于谁：
    #   · 宠物本体 ⇒ `<根>/<id>/meta.tsv` ✓（帧和它同层 ✓）；
    #   · **宠物装备** ⇒ `<根>/<装备id>/meta.tsv` ✓（帧在 `<装备id>/<宠物id>/` ✓ **上一层** ✗）
    #     —— 而且那里的行名**带 `<宠物id>/` 前缀**（见 `WzProbe/PetDump.cs` ✓）
    #     ⇒ 不这么找，装备帧的 origin 会**全丢** ✗ ⇒ 组合叠歪（实测抓到的就是这个 ✓）。
    meta = _parse_meta(d / "meta.tsv")
    if not meta and pet_id is not None:
        meta = _parse_meta(Path(root) / str(sid) / "meta.tsv")
    got = []
    for f in d.iterdir():
        if not f.is_file() or not f.name.lower().endswith(".png"):
            continue
        stem = f.stem
        if not stem.startswith(str(state) + "_"):
            continue
        tail = stem[len(str(state)) + 1:]
        if not tail.isdigit():
            continue
        # ⚠ 装备那边的行名带 `<宠物id>/` 前缀 ⇒ **两个键都试** ✓（见上面那段说明 ✓）
        ox, oy, _dl = meta.get(f.name, meta.get("%s/%s" % (pet_id, f.name),
                                                (None, None, 0.0)))
        got.append((int(tail), f, ox, oy))
    got.sort(key=lambda t: t[0])
    return [(f, ox, oy) for _n, f, ox, oy in got]


def sprite_icon(root, sid):
    """列表用的那张小图（`icon.png` ✓）；没有 ⇒ `None` ✓（界面就画空图标 ✓ 不回退 ✗）。"""
    p = Path(root) / str(sid) / "icon.png"
    return p if p.is_file() else None


# ══════════════════════════════════════════════════════════════
# 组合外观模板（方案 C ✓）
# ══════════════════════════════════════════════════════════════
def norm_pets(raw):
    """把卡片里存的那张列表归一成 `[{"pet", "name", "equips"}]` ✓（与 `wzexport.norm_drops` 对称 ✓）。

    容错三种写法（老配置/手写的都能吃 ✓）：
      · `{"pet": "5000000", "name": "褐色小猫", "equips": ["01802000"]}` ✓（卡片存的就是它 ✓）
      · `{"id": "5000000", "equips": [...]}` ✓（把 `id` 当宠物 id ✓）
      · `"5000000"` ✓（只有 id ⇒ 只标本体、不戴装备 ✓）
    ⚠ 同一只宠物**可以有多套**（戴不同装备 ✓）⇒ **不去重**（`equips` 不同的两条都要留着 ✓）；
      但同一只 + **同一组装备**重复出现 ⇒ 去掉 ✓（手改配置很容易写成这样 ✓）。
    """
    out, seen = [], set()
    for d in (raw or []):
        if isinstance(d, dict):
            pid = str(d.get("pet") or d.get("id") or "").strip()
            eq = sorted({str(x).strip() for x in (d.get("equips") or []) if str(x).strip()})
            nm = str(d.get("name") or "").strip()
        else:
            pid, eq, nm = str(d or "").strip(), [], ""
        if not pid:
            continue
        key = (pid, tuple(eq))
        if key in seen:
            continue
        seen.add(key)
        out.append({"pet": pid, "name": nm, "equips": eq})
    return out


def combo_key(pet_id, equip_ids=()):
    """组合的**目录名**（可复现 ✓）：`<宠物id>__<装备id>+<装备id>…`（装备按 id 排序 ✓，
    顺序不同但集合相同 ⇒ 同一个目录 ✓ 不重复做 ✗）。"""
    eq = "+".join(sorted(str(x).strip() for x in (equip_ids or []) if str(x).strip()))
    return "%s__%s" % (str(pet_id), eq) if eq else str(pet_id)


def combo_root():
    """组合模板放哪儿 —— ⚠ **不进图库目录**（那是导出的东西、随时可能被重导覆盖 ✗），
    单独放 `datasets/sprites/pet_combo/` ✓（由我们生成、可随时删掉重做 ✓）。"""
    return _cfg_path("pet_combo_dir", "datasets/sprites/pet_combo")


# ══════════════════════════════════════════════════════════════
# 组合要用的图**导没导**（用户 2026-10-05 ✓）
# ══════════════════════════════════════════════════════════════
# 为什么要这一块（用户 2026-10-05 报的现场 ✓ 原话："给要标注的宠物配上装备后，在背包窗格中
#   双击，弹窗里显示要标注的帧发现除了 stand 动画全都没有装备（鳄鱼潭1，花蘑菇仔）"）：
#   · 组合是**按姿态分别叠**的 ✓（`compose_pet`：每个 state 去装备库取该姿态的帧，
#     **取不到就静默跳过那一件装备** ✗）⇒ 界面显示的没错，它照实画了"没装备的组合" ✓；
#   · 真因在**图库**：全库 **52 件装备都只有 `stand0`** ✗（2026-10-05 实测逐个扫过 ✓），
#     而本体有 `stand0/stand1/move/jump/hungry/sit/angry` ✓ ——
#     源头是导出器 `WzProbe dump-petequips` 的 `--states` **默认 `stand0`** ✓
#     （`PetDump.cs::RunPetEquips` ✓ 那时组合还只做 stand0 ⇒ 两边漂了 ✓）。
# ⇒ 所以这里给界面一个**判据**："这套组合要用的图，哪些还没导" ✓ —— 好让它**当场读条补导** ✓
#   （用户 2026-10-05 ✓ 原话："如果没有导出，在添加的带装备的宠物、关闭弹窗后，在数据集工作台
#   显式读条导出" ✓），而不是让人自己去猜"为什么没有装备" ✓。


def combo_state_names(states=None):
    """组合外观**会做**哪些姿态（默认 = `COMBO_STATES` ✓）—— 口径只此一处 ✓。"""
    return tuple(states or COMBO_STATES)


def _states_on_disk(d, states):
    """目录 `d` 里**已经有帧**的姿态（只看文件名 ✓ 不读 meta ✓ 快 ✓）。"""
    have = set()
    d = Path(d)
    if d.is_dir():
        for f in d.glob("*.png"):
            if f.stem == "icon":              # 图标不是帧 ✓（`icon.png` 会把 stem 算成 "icon" ✓）
                continue
            st = f.stem.rsplit("_", 1)[0]     # `move_3` ⇒ `move` ✓（姿态名里没有下划线 ✓）
            if st in states:
                have.add(st)
    return have


def equip_state_gaps(pets, states=None):
    """要标注的装备**哪些姿态还没有帧** ⇒ `{装备id: [姿态…]}`（空表 = 都齐了 ✓）。

    ⚠ 口径看的是 **(装备, 宠物)**（装备的美术按宠物分开存 ✓ 见 `sprite_frames` ✓），
      但导出是**按装备**跑一趟、会带上它所有能戴的宠物 ✓ ⇒ 这里按装备**取并集** ✓
      （不然"这件装备对 A 宠物缺 `move`、对 B 宠物缺 `jump`"会被拆成两趟 ⇒ 白跑一遍 ✓）。
    """
    states = combo_state_names(states)
    root = pet_equip_dir()
    out = {}
    for d in (pets or []):
        d = d or {}
        pid = str(d.get("pet") or d.get("id") or "").strip()
        if not pid:
            continue
        for e in (d.get("equips") or []):
            e = str(e).strip()
            if not e:
                continue
            have = _states_on_disk(root / e / pid, states)
            miss = [s for s in states if s not in have]
            if miss:
                cur = out.setdefault(e, [])
                for s in miss:
                    if s not in cur:
                        cur.append(s)
    for e in out:                             # 顺序按 `COMBO_STATES` ✓（命令里列出来稳定 ✓）
        out[e] = [s for s in states if s in out[e]]
    return out


def pet_state_gaps(pets, states=None):
    """**本体**缺姿态的宠物 id ⇒ `[宠物id…]`（`dump-pets` 一趟会把它所有姿态全导 ✓）。"""
    states = combo_state_names(states)
    root = pet_dir()
    out = []
    for d in (pets or []):
        d = d or {}
        pid = str(d.get("pet") or d.get("id") or "").strip()
        if not pid or pid in out:
            continue
        have = _states_on_disk(root / pid, states)
        if any(s not in have for s in states):
            out.append(pid)
    return out


def combo_export_plan(pets, states=None):
    """**要补导什么** ⇒ `[{"cmd": "dump-pets"|"dump-petequips", "ids": [...], "states": [...]}, …]`。

    空表 = 都齐了 ⇒ 不用导 ✓（工作台那边据此决定要不要读条 ✓ 见 `gui/steps/cards.py` ✓）。
    ⚠ `dump-pets` **没有 `--states`**（`PetDump.RunPets` 一趟全导 ✓）⇒ 给它的 `states` 留空表 ✓
      —— 拼命令的人（`core/wzexport.pet_export_task` ✓）别给本体也塞 `--states` ✗。
    """
    states = combo_state_names(states)
    plan = []
    pids = pet_state_gaps(pets, states)
    if pids:
        plan.append({"cmd": "dump-pets", "ids": pids, "states": []})
    gaps = equip_state_gaps(pets, states)
    if gaps:
        # 装备那条的 `--states` 是**一件装备一份**（`RunPetEquips` ✓）⇒ 缺的姿态不同的装备
        # 要分开跑 ✓（绝大多数情况一模一样 ⇒ 只有一趟 ✓）。
        by_key = {}
        for e in sorted(gaps):
            by_key.setdefault(tuple(gaps[e]), []).append(e)
        for st, ids in by_key.items():
            plan.append({"cmd": "dump-petequips", "ids": ids, "states": list(st)})
    return plan


def _stamp(pet_id, equip_ids):
    """组合的**源指纹**（本体/装备的目录 mtime+size ✓）—— 源一变就重做 ✓。"""
    parts = [str(pet_dir() / str(pet_id)), str(wzexport._dir_stamp(pet_dir() / str(pet_id)))]
    for e in sorted(equip_ids or []):
        p = pet_equip_dir() / str(e) / str(pet_id)     # ⚠ 装备帧在**这只宠物**那一层 ✓

        parts += [str(p), str(wzexport._dir_stamp(p))]
    return tuple(parts)


def _read_rgba(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _over(dst, src, x, y):
    """把 `src` 按 alpha **叠**到 `dst` 的 (x, y)（越界就裁 ✓ —— 少几个像素也别抛 ✗）。"""
    h, w = src.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + w), min(dh, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = src[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = dst[y0:y1, x0:x1]
    if sub.shape[2] == 4:
        a = (sub[:, :, 3:4].astype(np.float32) / 255.0)
        roi[:, :, :3] = (sub[:, :, :3].astype(np.float32) * a
                         + roi[:, :, :3].astype(np.float32) * (1.0 - a)).astype(np.uint8)
        roi[:, :, 3:4] = np.maximum(roi[:, :, 3:4], sub[:, :, 3:4])
    else:                                   # 没有 alpha 通道 ⇒ 直接盖 ✓
        roi[:] = sub


def compose_pet(pet_id, equip_ids=(), state=None, force=False, states=None):
    """把「宠物 + 装备」按 origin 对齐叠成**组合外观模板** ⇒ 返回输出目录 ✓（方案 C ✓）。

    产出（**与怪物/掉落精灵库同一套格式** ✓ 见模块说明）::

        <combo_root>/<combo_key>/
            stand0_0.png … move_3.png      组合帧（各帧按 origin 对齐叠好 ✓）
            meta.tsv                       与预览同格式（`#file originX originY …` ✓）
            meta.json                      {"pet": …, "equips": [...], "states": [...]}（界面回读 ✓）

    ⭐⭐ **默认做 `COMBO_STATES` 那几个姿态**（用户 2026-10-04 ✓ 原话："多姿态进组合外观" ✓）：
      实测只做 `stand0` ⇒ 1 框/8% ✗；加上 `move` ⇒ **8 框/42%** ✓✓（见那个常量的说明 ✓）。
      ⚠ 管线那边**不用改** ✓：`detect_mobs.load_templates` 是 `*.png` 通吃的 ✓（一棵目录里
        有几套姿态就吃几套 ✓，上限走 `per_mob` ✓）。
    ⚠ 参数：`states=(…)` ⇒ 只做这几个 ✓；`state="move"` ⇒ 只做**这一个**（老调用方 / 单姿态
      预览用 ✓ 行为与老版一字不差 ✓）；两个都不给 ⇒ 走 `COMBO_STATES` ✓。

    返回 `None` = **做不出来**（宠物这几个姿态一帧都没有 / 图读不了 ✓）—— 调用方**如实说** ✓ 别静默 ✗。

    ⚠ 缺帧的口径：装备帧数比宠物少时**按 `i % len(装备帧)` 取** ✓（见模块说明 ✓）。
    ⚠ 缓存：源目录指纹（mtime/size ✓）没变**且**这一组姿态的产物在 ⇒ 不重做 ✓；
      缓存键里**带姿态列表** ✓ ⇒ 老版本只做 stand0 的那份缓存**不会被误用** ✓（会重做 ✓）。
    ⚠ 每次真做之前**先清掉目录里的旧 png** ✗➡✓：姿态列表哪天变少（用户改了/倒退），
      残留的旧帧会**照样被当模板**（管线是通吃的 ✓）⇒ 那就是"标不到还白算" ✓。
    """
    pet_id = str(pet_id or "").strip()
    equip_ids = [str(x).strip() for x in (equip_ids or []) if str(x).strip()]
    if states is None:
        states = (state,) if state else tuple(COMBO_STATES)
    states = tuple(str(s).strip() for s in states if str(s).strip()) or (DEFAULT_STATE,)
    out_dir = combo_root() / combo_key(pet_id, equip_ids)
    key = (pet_id, tuple(sorted(equip_ids)), states)
    stamp = _stamp(pet_id, equip_ids)
    if (not force and _COMBO_STAMP.get(key) == stamp
            and (out_dir / "meta.tsv").is_file()):
        return out_dir

    per_state, metas = {}, []                 # metas 跨姿态累积（写一份 meta.tsv ✓）
    first_write = True
    for st in states:
        base = sprite_frames(pet_dir(), pet_id, st)
        if not base:
            continue                          # 这只宠物没有这个姿态 ⇒ 跳过它 ✓（别的还有 ✓）
        layers = []
        for e in equip_ids:
            # ⚠ 装备要**按这只宠物**取帧（`<装备>/<宠物id>/` ✓ 见 `sprite_frames` ✓）：
            #   没有那个子目录 ⇒ 这只宠物戴不了它 ⇒ **跳过** ✓（不至于因为一件戴不上的装备
            #   就把整只组合弄失败 ✗；界面本来就只列"能戴的" ✓ 见 `pet_equips.json` 的 `pets` ✓）。
            fs = sprite_frames(pet_equip_dir(), e, st, pet_id=pet_id)
            if fs:
                layers.append(fs)

        made_here = 0
        for i, (pf, pox, poy) in enumerate(base):
            pimg = _read_rgba(pf)
            if pimg is None:
                continue
            if pox is None or poy is None:   # meta 缺这一帧 ⇒ 退回"以中心为原点"✓（同 cutpaste ✓）
                pox, poy = pimg.shape[1] // 2, pimg.shape[0]
            drawn = [(pimg, float(pox), float(poy))]
            for fs in layers:
                ef, eox, eoy = fs[i % len(fs)]
                eimg = _read_rgba(ef)
                if eimg is None:
                    continue
                if eox is None or eoy is None:
                    eox, eoy = eimg.shape[1] // 2, eimg.shape[0]
                drawn.append((eimg, float(eox), float(eoy)))
            # ⚠ 这段是**以 origin 为原点**算包围盒：某张图画在 x ∈ [-ox, -ox+w] ✓
            #   （生成式里的元组顺序别抄错 ✗ —— 本轮就写错过一次：`_img, _ox, oy` 里又用 `img`）
            xs0 = min(-ox for _img, ox, _oy in drawn)
            ys0 = min(-oy for _img, _ox, oy in drawn)
            xs1 = max(-ox + im.shape[1] for im, ox, _oy in drawn)
            ys1 = max(-oy + im.shape[0] for im, _ox, oy in drawn)
            w, h = int(round(xs1 - xs0)), int(round(ys1 - ys0))
            if w <= 0 or h <= 0:
                continue
            canvas = np.zeros((h, w, 4), np.uint8)
            for img, ox, oy in drawn:        # 宠物先、装备后 ✓（绘制顺序同 Pet.cs ✓）
                _over(canvas, img, int(round(-ox - xs0)), int(round(-oy - ys0)))
            if first_write:
                out_dir.mkdir(parents=True, exist_ok=True)
                # ⚠ 清掉上一次留下的 png（姿态列表变了 ⇒ 旧帧会**照样被当模板** ✗ 见上面说明 ✓）；
                #   meta.* 也一起清 ✓（下面会重写 ✓）。
                for old in list(out_dir.glob("*.png")) + list(out_dir.glob("meta.*")):
                    try:
                        old.unlink()
                    except OSError:
                        pass
                first_write = False
            name = "%s_%d.png" % (st, i)
            if not cv2.imwrite(str(out_dir / name), canvas):
                return None
            metas.append((name, int(round(-xs0)), int(round(-ys0))))
            made_here += 1
        if made_here:
            per_state[st] = made_here

    if not metas:
        return None
    lines = ["#file\toriginX\toriginY\tltX\tltY\trbX\trbY\tdelay"]
    for name, ox, oy in metas:
        lines.append("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d"
                     % (name, ox, oy, -ox, -oy, 0, 0, 0))
    (out_dir / "meta.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"pet": pet_id, "equips": equip_ids, "states": list(per_state),
                    "per_state": per_state, "frames": len(metas)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _COMBO_STAMP[key] = stamp
    # ⚠⚠ 目录是**按「宠物+装备」共享的**（`combo_key` 不含姿态 ✗）⇒ 这次只写了 `states`
    #   这几个姿态 ✓ ⇒ **同一只的其它缓存键必须作废** ✗（否则"先收窄成 stand0、再切回默认"
    #   会命中旧键、直接返回那个只有 stand0 的目录 ✗ —— 用例 `t_combo_multi_states` 抓到的 ✓）。
    for k in [k for k in list(_COMBO_STAMP)
              if k[0] == pet_id and k[1] == tuple(sorted(equip_ids)) and k != key]:
        _COMBO_STAMP.pop(k, None)
    return out_dir


def drop_combo(pet_id, equip_ids=()):
    """删掉某个组合的产物（改了装备/重导之后想强制重做时用 ✓）；没有 ⇒ 也算成功 ✓。

    ⚠ 缓存键里现在**带姿态列表**（`compose_pet` 的 `key` ✓）⇒ 这里要把这只宠物+这组装备的
      **所有**键都清掉 ✗➡✓（只清 `DEFAULT_STATE` 那个的话，改了姿态集合时旧键还在 ✓
      ⇒ 下次照样走缓存、白改 ✓）。
    """
    out_dir = combo_root() / combo_key(pet_id, equip_ids)
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    pid, eq = str(pet_id), tuple(sorted(str(x) for x in (equip_ids or [])))
    for k in [k for k in list(_COMBO_STAMP) if k[0] == pid and k[1] == eq]:
        _COMBO_STAMP.pop(k, None)
    return not out_dir.exists()


def combos_state():
    """已经做好的组合目录 → `[目录名, …]`（给界面/自检查看 ✓ 只读 ✓）。"""
    root = combo_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in os.scandir(str(root)) if p.is_dir())
