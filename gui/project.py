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
    # 用户既算不出来也用不上，界面上不显示它。
    "scale": 1.12,
    # 标定时的画面信息，用来判断"画面变了吗、要不要重标"
    "scale_at": {},       # {width, height, mob, confidence}
    "calib": {           # ③ 标定的扫描范围
        "min_scale": 0.4,
        "max_scale": 2.4,
        "step": 0.10,
        "sample": 5,     # 采样几帧做标定（单帧可能正好没有目标）
    },
    "notes": "",
    "capture": {
        "source": "window",   # window | file | stream
        "file": "",
        "url": "udp://0.0.0.0:5000",
        "fps": 5.0,           # 窗口捕获频率
        "stride": 6,          # 文件抽帧间隔
        "seconds": 120.0,
        "dedup": 0.06,        # 实测：低于 0.06 基本滤不掉任何东西
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
    """递归合并，保证旧项目文件缺字段时能补上默认值。"""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
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
        """取一个配置段（capture / label / dataset / train）。"""
        return self.data.get(name, {})

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
