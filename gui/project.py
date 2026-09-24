"""项目 = 一张地图。所有产物自包含在项目目录里。

    projects/101040001_野猪领土/
        project.yaml        <- 全部配置
        frames/             <- ② 抽帧产物
        calib/              <- ③ 尺度标定结果 + 对比图
        labels_auto/        <- ④ 自动标注原始结果（只读，永不修改）
        labels/             <- ⑤ 编辑器写入区（人工修正后的标注）
        labels_backup/      <- 每次保存前的旧版本（时间戳），防手抖
        labels_iter/        <- ⑨ 迭代自训练的产物，优先级介于上面两者之间
        vis/                <- ④ 可视化图，供质检台翻看
        dataset/            <- ⑥ YOLO 格式
        runs/  models/      <- ⑦ 训练输出
        verify/             <- ⑧ 推理结果图 + 统计

为什么 labels_auto 和 labels 分开：
    任何时刻都能回退到「纯自动标注」的状态，方便对比
    「人工修过的集」和「没修过的集」训出来的模型差别。
"""

import copy
import json
import time
from pathlib import Path

import yaml

SUBDIRS = (
    "frames", "calib",
    "labels_auto", "labels", "labels_backup", "labels_iter",
    "vis", "dataset", "runs", "models", "verify",
)

DEFAULTS = {
    "name": "",
    "map_id": "",
    "mobs": [],          # 本图出现的怪种 id 列表
    # ③ 标定结果。这是**内部中间量**：取决于游戏分辨率、窗口大小、推流缩放，
    # 用户既算不出来也用不上，界面上不显示它。None = 还没标定。
    "scale": None,
    # 逐怪/逐动作的尺度覆盖：{mob_id: scale, "mob_id:action": scale}。
    # 手动目测标定时写入，标注时按「怪+动作 → 怪 → scale」三级回退。
    "mob_scales": {},
    # 标定时的画面信息，用来判断"画面变了吗、要不要重标"
    "scale_at": {},       # {width, height, mob, confidence}
    "calib": {           # ③ 标定的扫描范围
        "min_scale": 0.4,
        "max_scale": 2.4,
        "step": 0.10,
        "sample": 5,     # 采样几帧做标定（单帧可能正好没有目标）
    },
    # HP/MP 条框选结果（相对画面的比例 [nx, ny, nw, nh]，分辨率变化自动适配）。
    # 存项目里 —— 不同项目可能用了不同的游戏 UI 布局；项目没存过的回退用
    # **最近打开的那个项目**的（= 你上一次框选的位置，见 last_opened）。
    "bars": {},
    # 整份决策参数（DecisionSettings.to_dict() 的结果，几十个键）：**按项目各存一份**。
    # 换项目就换一套（攻击距离、序列、定时行为、防掉线……都跟着项目走）—— 这也是
    # 「一开项目就是另一个项目的参数」这个老毛病的解药：参数只认项目，没有全局副本。
    # 空的 = 这个项目还没存过：打开时拿**最近打开的那个项目**那份播种并当场写回来
    # （省得键位/序列/血条重配一遍），之后各项目互不影响。
    "decision": {},
    "notes": "",
    "capture": {
        "source": "window",   # window | file | stream
        "file": "",
        "url": "udp://0.0.0.0:5000",
        "fps": 5.0,           # 窗口捕获频率
        "stride": 6,          # 文件抽帧间隔
        "seconds": 120.0,
        "dedup": 0.01,        # 变化像素比例阈值：低于它视为重复帧
        "limit": 0,
        "clean": True,        # 重抽前清空旧帧，避免新旧混合
    },
    "label": {
        "thresh": 0.90,
        "min_distinct": 0.06,
        # 模板帧**上限**：非死亡帧数不超过它时全部使用（常规情况），
        # 超过才按动作轮询取样。曾经设成 6，把绿水灵 stand 的第三个相位
        # 挤掉了，导致那个相位的怪从来没被标注过。
        "per_mob": 20,
        "max_peaks": 3,
        "downscale": 1,
        "region": None,      # "x,y,w,h"
    },
    "dataset": {
        "val_ratio": 0.2,
        "min_boxes": 1,
        "quality": 92,
    },
    "train": {
        "model": "yolo26n.pt",
        "epochs": 120,
        "imgsz": 960,
        "batch": 8,
        "device": "0",
        "patience": 40,
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    """递归合并：以 base（模块级 DEFAULTS）为底，用 over 覆盖。

    返回**全新对象**（base 深拷贝，over 的取值也深拷贝）。

    这里必须深拷贝：原来只做 `dict(base)` 浅拷贝，凡是 over 里没有的嵌套
    dict/list 都会和 DEFAULTS 共享同一个对象。而 cards.py 里到处是
    `project.sec("train").update(...)` 这种原地修改 —— 一改就写穿到
    DEFAULTS，污染之后新建 / 打开的所有项目（表现为「新项目莫名继承了
    上一个项目调过的参数」）。旧版本的 project.yaml 缺新加的键，最容易中招。
    """
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Project:
    def __init__(self, root):
        # 必须存绝对路径。相对路径（比如 "projects/某图"）传给 ultralytics
        # 会被当成本地相对目录，拼到默认的 runs/detect/ 后面 ——
        # 训练结果就跑到项目外面去了，而且不报错。
        self.root = Path(root).resolve()
        self.path = self.root / "project.yaml"
        self.data = _deep_merge(DEFAULTS, {})

    # ---------------- 创建 / 打开 / 保存 ----------------

    @classmethod
    def create(cls, root, name="", map_id=""):
        p = cls(root)
        p.root.mkdir(parents=True, exist_ok=True)
        for d in SUBDIRS:
            (p.root / d).mkdir(exist_ok=True)
        p.data["name"] = name or p.root.name
        p.data["map_id"] = map_id
        p.save()
        return p

    @classmethod
    def open(cls, root):
        p = cls(root)
        if not p.path.exists():
            raise FileNotFoundError("不是项目目录（缺 project.yaml）: %s" % root)
        with open(p.path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        p.data = _deep_merge(DEFAULTS, raw)
        for d in SUBDIRS:
            (p.root / d).mkdir(exist_ok=True)
        return p

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, allow_unicode=True,
                           sort_keys=False, default_flow_style=False)

    # ---------------- 取值 ----------------

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val, save=False):
        self.data[key] = val
        if save:
            self.save()

    def sec(self, name):
        """取一个配置段；不存在则创建（返回 self.data 里的引用，改它能被 save 保存）。"""
        if name not in self.data:
            self.data[name] = {}
        return self.data[name]

    def dir_of(self, name):
        d = self.root / name
        d.mkdir(exist_ok=True)
        return d

    @property
    def frames(self):
        return self.dir_of("frames")

    @property
    def calib(self):
        return self.dir_of("calib")

    @property
    def vis(self):
        return self.dir_of("vis")

    @property
    def dataset(self):
        return self.dir_of("dataset")

    # ---------------- 产物清点（供界面显示） ----------------

    def count(self, which):
        """数一个目录里的文件。which: frames/*.png 用 'frames' 这类名字。"""
        d = self.root / which
        if not d.is_dir():
            return 0
        return sum(1 for _ in d.iterdir() if _.is_file())

    def snapshot(self):
        """当前项目的产物概况，界面用它渲染卡片底部摘要。"""
        return {
            "frames": sum(1 for _ in self.frames.glob("*.png")),
            "labels_auto": sum(1 for _ in self.dir_of("labels_auto").glob("*.txt")),
            "labels": sum(1 for _ in self.dir_of("labels").glob("*.txt")),
            "vis": sum(1 for _ in self.vis.glob("*.jpg")),
            "mobs": len(self.data.get("mobs") or []),
            "scale": self.data.get("scale"),
            "has_dataset": (self.dataset / "data.yaml").exists(),
            "has_model": any(self.dir_of("models").glob("*.pt")),
        }


def sanitize(name: str) -> str:
    """把地图名变成安全的目录名。"""
    bad = '<>:"/\\|?*'
    for c in bad:
        name = name.replace(c, "_")
    return name.strip().strip(".") or ("project_" + time.strftime("%Y%m%d_%H%M%S"))


# ══════════════════════════════════════════════════════════════
# 最近打开的项目（工作台自己的状态）
# ══════════════════════════════════════════════════════════════
#
# **为什么要记住它**：决策参数只按项目存了，那「还没打开项目」的时候用谁的？
# 用「最近打开的那个」：参数面板一进来就是你上次调好的那套，而不是一份飘忽的
# 全局副本（旧版 config/decision.json 就是那么干的 —— 谁改参数就被谁覆盖，
# 于是新项目一打开常常是另一个项目的参数）。
#
# 存在仓库 config/ 下，不存在项目里：这是**界面自己的**状态，与项目内容无关；
# 放进项目里会跟着项目一起被拷到别的机器，语义就乱了。
SESSION_FILE = Path(__file__).resolve().parent.parent / "config" / "session.json"


def remember_open(root):
    """记下「最近打开的项目」—— 打开 / 新建项目成功后调用。"""
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(
            json.dumps({"last_project": str(Path(root).resolve())},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    except Exception:
        pass      # 记不住只会退化成「没有最近项目」，不该影响打开项目本身


def last_opened():
    """最近打开的那个项目（Project 或 None）。

    项目被删 / 改名 / 文件坏了都返回 None，调用方一律当「没有最近项目」处理
    （用默认值），不要因为一个陈旧的路径就报错。
    """
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        root = (data or {}).get("last_project")
    except Exception:
        return None
    if not root:
        return None
    try:
        return Project.open(root)
    except Exception:
        return None
