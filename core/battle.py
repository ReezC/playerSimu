"""「战斗区域」（battle_zones）按**地图 id** 的持久化。

2026-10-01 起从「项目（project.yaml 的 decision.battle_zones）」搬到**按地图 id 存**：
`datasets/map/<id>.battle.json`（和 `zones.json` / `.mapcalib.json` 并列 ✓）—— 战斗区域
每一项都是**地图里的东西**（`set` = 本图集合名、`idle_footholds` = 本图 foothold 编号、
`fight_dst` = 又是集合名 ✓），换图就废 ⇒ 跟着地图走 ✓。

⚠ **这里只做「读文件 / 写文件 / 路径」**，**不碰清洗**：清洗口径仍在
`decision.agent.DecisionSettings._load_battle_zones`（**一处** ✓）—— 那是"手改坏的 json
不该被带进运行期"的判据（去重 / 数值钳位 / 老键迁移），别在 core 里再写一份 ✗。
写文件也**只写已经清洗过的列表**（调用方负责 ✓）。

⚠ 分层：core（第 0 层）不许 import decision ⇒ 本模块不引用 `DecisionSettings` ✗。
"""
import json

from core import mapdata


def path(map_id):
    """这张图的战斗区域文件：`datasets/map/<id>.battle.json`。"""
    return mapdata.map_dir() / ("%s.battle.json" % map_id)


def load(map_id):
    """读某张图的战斗区域 → `list | None`。

    ⚠ `None` = **文件不存在**（= 这张图还没配过 / 要从旧 project.yaml 迁移 ✓）；
       `[]` = 文件在但**空列表**（= 明确"不限制" ✓）。调用方要靠这个区别决定要不要迁移 ✗
       —— 别把"没配过"和"配了但清空了"混成一件事 ✗。
    """
    p = path(map_id)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                        # noqa: BLE001 —— 坏了就当"没这份"，别崩 ✗
        return None
    if isinstance(raw, dict):
        return raw.get("battle_zones")
    if isinstance(raw, list):
        return raw
    return None


def save(map_id, zones):
    """把（**已经清洗过的**）战斗区域列表写进 per-map 文件。"""
    p = path(map_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"battle_zones": [dict(z) for z in zones]},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    return p
