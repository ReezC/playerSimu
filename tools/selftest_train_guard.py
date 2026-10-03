"""自检：**训练链路的两个守卫**（2026-10-03 新增 ✓ 用户现场"怎么没有检出框了"）。

为什么要有它 —— 这条链**出错时一句错都不报**，最后只表现为"**实时一个框都没有**"✗：

  · **「接着上次跑」在续不了的时候会静默退化**（ultralytics 只打一句 warning，然后当成
    普通训练跑 ✗）—— 而 resume 那一路**只传 `resume`** ⇒ 它用**自带的默认数据集
    `coco8.yaml`**（还联网下载 COCO ✗）、`project=null` ⇒ 产物落到 `runs/detect/train/`、
    `epochs=100`（默认）⇒ 100 轮训出个 **80 类 COCO 模型**（mAP 全 0 ✓），训完还**被发布成
    项目正式权重** ✗ ⇒ 实时那边"按 mtime 选最新"选中它 ⇒ **一个框都不认** ✓。
    现场（森林迷宫III → 石人寺院III 的续训）就是这么发生的 ✓。
  · **发布前不看类名**：一份"和本项目无关"的权重会被写成 `models/detect_vN.pt`（= 正式权重 ✗），
    而且**覆盖掉上一版好的** —— 事后没人能从文件名看出来 ✗。

⇒ 本自检钉四件（都走**真路径** ✓ 用假的 `ultralytics` 注入，不真训练 ✓）：
  ① `ckpt_resumable`：`epoch < 0` / 没优化器 ⇒ 不能续（**这是"上一次正常跑完"的常态** ✓）；
  ② **续不了 ⇒ 报错停下**，而且**连模型都不加载**（钉"绝不让它退化成普通训练" ✓）；
  ③ 真续训时，`data=` / `project=` / `name=` **也一起传**（万一它又退化，用的还是咱的数据 ✓）；
  ④ **发布前对类名**：对不上 ⇒ **不发布**，而且 `models/` 里**上一版必须原封不动** ✓；
  ⑤ 源码级：实时那侧"开始推理"时会校验 `player`/`mob`，缺了就**当场告警**（弹窗 + perf 打点 ✓）。

跑法：
    python -m tools.selftest_train_guard
"""

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认是 GBK：用例名里带 ⇒ / ✓ 这类字符时 `print` 会抛 UnicodeEncodeError
# —— 那和被测的东西无关 ✓（同 `tools/selftest_mmap_regions.py` 的处理 ✓）。
try:
    sys.stdout.reconfigure(encoding="utf-8")            # type: ignore[attr-defined]
except Exception:                                       # noqa: BLE001
    pass

from core.context import CollectContext                 # noqa: E402
from perception import train as T                       # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


_TMP = None


def _tmp():
    global _TMP
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="selftest-trainguard-"))
    return _TMP


#: 本项目正常的 5 类（顺序就是类 id ✓ —— 顺序错了等于张冠李戴 ✗）
GOOD_NAMES = {0: "player", 1: "mob", 2: "drop", 3: "npc", 4: "other_player"}
#: 现场那份坏权重的类名（ultralytics 自带的 coco8 ⇒ 80 类 COCO ✓）
BAD_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}


def _save_ckpt(path, epoch, optimizer, names=GOOD_NAMES):
    """造一个"像 ultralytics 检查点"的文件（`epoch` / `optimizer` 是 `resume` 要看的两样 ✓）。"""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "optimizer": optimizer,
                "model": SimpleNamespace(names=dict(names))}, str(path))
    return path


def _data_yaml(root, names=GOOD_NAMES):
    """造一份 data.yaml（`names` 有序 ✓ —— `run_train` 发布前就是拿它对类名 ✓）。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "data.yaml"
    body = "path: %s\ntrain: images/train\nval: images/val\n\nnames:\n%s" % (
        root.as_posix().replace("\\", "/"),
        "".join("  %d: %s\n" % (k, names[k]) for k in sorted(names)))
    p.write_text(body, encoding="utf-8")
    return p


class _FakeYOLO:
    """假的 `ultralytics.YOLO`：只记录**被怎么调的**，并按要求造出 best.pt。"""

    made = []          # 每次构造（钉"续不了时连模型都不该加载" ✓）
    calls = []         # 每次 train(**kw)
    names = GOOD_NAMES  # 这次"训练"产出的权重类名（用例自己摆 ✓）

    def __init__(self, w):
        _FakeYOLO.made.append(str(w))
        self.w = str(w)
        self.trainer = SimpleNamespace(metrics={})
        self._cbs = {}

    def add_callback(self, name, fn):
        self._cbs[name] = fn

    def train(self, **kw):
        _FakeYOLO.calls.append(kw)
        save_dir = Path(str(kw.get("project") or "runs/detect")) / str(kw.get("name") or "train")
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_ckpt(save_dir / "weights" / "best.pt", 7, None, names=_FakeYOLO.names)
        for _n, fn in self._cbs.items():
            if _n == "on_train_end":
                fn(SimpleNamespace(save_dir=str(save_dir)))
        self.trainer = SimpleNamespace(metrics={"metrics/mAP50(B)": 0.9})


def _with_fake_ultralytics(fn):
    """注入假 ultralytics（`run_train` 里是函数内 `from ultralytics import YOLO` ✓）。"""
    _FakeYOLO.made, _FakeYOLO.calls = [], []
    old = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = SimpleNamespace(YOLO=_FakeYOLO)
    try:
        return fn()
    finally:
        if old is None:
            sys.modules.pop("ultralytics", None)
        else:
            sys.modules["ultralytics"] = old


def _project(root, names=GOOD_NAMES, resume_ok=True):
    """一个像项目的临时目录：data.yaml + 一份 last.pt（可续 / 不可续由 `resume_ok` 决定 ✓）。"""
    root = Path(root)
    data = _data_yaml(root / "dataset", names)
    last = _save_ckpt(root / "runs" / "detect_v3" / "weights" / "last.pt",
                      (5 if resume_ok else -1),
                      ({"state": {}} if resume_ok else None))
    (last.parent.parent / "args.yaml").write_text(
        "imgsz: 1280\nbatch: 8\nepochs: 120\nname: detect_v3\n", encoding="utf-8")
    return root, data, last


def t_resumable_reads_epoch_and_optimizer():
    """① `ckpt_resumable`：只认 `epoch`（>=0）+ 优化器状态在位 —— 就是 ultralytics 要读的两样。"""
    try:
        import torch                                            # noqa: F401
    except ImportError:
        check(True, "跳过（没装 torch）")
        return
    d = _tmp() / "r1"
    ok_bad = _save_ckpt(d / "a.pt", -1, None)                   # 上一次**正常跑完**的 last.pt
    ok_good = _save_ckpt(d / "b.pt", 5, {"state": {}})          # 崩掉/取消留下的 ✓
    no_opt = _save_ckpt(d / "c.pt", 5, None)
    broken = d / "d.pt"
    broken.write_text("这不是权重", encoding="utf-8")

    r, why = T.ckpt_resumable(ok_bad)
    check(r is False and why, "`epoch=-1` 的检查点被判成可续了（那会让 ultralytics 静默退化 ✗）")
    check("跑完" in why or "epoch" in why, "理由没说清（人要看得懂 ✗）：%r" % why)
    check(T.ckpt_resumable(ok_good)[0] is True, "带优化器状态的检查点被判成不能续 ✗")
    check(T.ckpt_resumable(no_opt)[0] is False, "没有优化器状态却判成能续 ✗")
    check(T.ckpt_resumable(broken)[0] is False, "坏文件该判「不能续」（而不是抛异常 ✗）")
    check(T.ckpt_resumable(d / "不存在.pt")[0] is False, "文件不存在时不该抛异常 ✗")


def t_not_resumable_stops_before_loading_model():
    """② 续不了 ⇒ 报错停下，而且**连模型都不加载**（钉"绝不退化成普通训练" ✗）。"""
    root = _tmp() / "r2"
    _proj, data, last = _project(root, resume_ok=False)
    ctx = CollectContext()

    def run():
        try:
            T.run_train({"data": str(data), "epochs": 120, "imgsz": 1280, "batch": 8,
                         "project_dir": str(root / "runs"), "name": "detect_v4",
                         "copy_to": str(root / "models"), "resume": True}, ctx)
        except RuntimeError as e:
            return str(e)
        return None

    msg = _with_fake_ultralytics(run)
    check(msg is not None,
          "「续不了」居然没报错（那就会退化成用 coco8 训练 ✗ —— 现场就是这么毁掉权重的 ✗）")
    check("续不了" in msg, "报错没直说「续不了」：%r" % msg[:120])
    check(last.as_posix() in msg and "best.pt" in msg,
          "报错里没给「该怎么办」（要指出基础权重该填哪一份 ✓）：%r" % msg[:220])
    check(_FakeYOLO.made == [],
          "续不了还去加载了模型（应该在加载之前就停下 ✗）：%r" % (_FakeYOLO.made,))
    check(_FakeYOLO.calls == [], "续不了还调了 train（这就是退化本身 ✗）：%r" % (_FakeYOLO.calls,))


def t_resume_also_passes_data_and_publishes_when_names_match():
    """③ 真续训：`resume=True` **也要带上** `data/project/name`（退化时的唯一保险 ✓）；
    类名对得上 ⇒ 正常发布 ✓。"""
    root = _tmp() / "r3"
    _proj, data, _last = _project(root, resume_ok=True)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    ctx = CollectContext()
    _with_fake_ultralytics(lambda: T.run_train(
        {"data": str(data), "epochs": 120, "imgsz": 1280, "batch": 8,
         "project_dir": str(root / "runs"), "name": "detect_v3",
         "copy_to": str(models), "resume": True}, ctx))
    check(len(_FakeYOLO.calls) == 1, "没真的续训：%r" % (_FakeYOLO.calls,))
    kw = _FakeYOLO.calls[0]
    check(kw.get("resume") is True, "没传 resume=True：%r" % (kw,))
    check(kw.get("data") == str(data),
          "续训没带 `data`（万一它退化成普通训练，就会用自带的 coco8 ✗）：%r" % (kw,))
    check(kw.get("project") and kw.get("name"),
          "续训没带 `project`/`name`（同样是为了「万一退化」时别写到默认目录 ✗）：%r" % (kw,))
    check((models / "detect_v3.pt").exists(),
          "类名对得上却没发布：%s" % sorted(p.name for p in models.glob("*")))


def t_publish_refuses_name_mismatch_and_keeps_old_weight():
    """④ **发布前对类名**：对不上就不发布，而且 `models/` 里**上一版必须原封不动** ✓。

    现场那件事：80 类 COCO 模型被写成 `models/detect_v3.pt`，**覆盖**了本项目的正常 v3 ✗
    ⇒ 实时按 mtime 选中它 ⇒ 0 框 ✓（本用例钉的就是"覆盖"这一步不许发生 ✓）。
    """
    root = _tmp() / "r4"
    _proj, data, _last = _project(root, resume_ok=True)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    old = models / "detect_v3.pt"
    old.write_bytes(b"OLD-GOOD-WEIGHT")            # 上一版正常权重（要保住 ✓）
    _FakeYOLO.names = BAD_NAMES                    # 这次"训"出来的是 COCO ✗
    ctx = CollectContext()
    try:
        _with_fake_ultralytics(lambda: T.run_train(
            {"data": str(data), "epochs": 120, "imgsz": 1280, "batch": 8,
             "project_dir": str(root / "runs"), "name": "detect_v3",
             "copy_to": str(models), "resume": False}, ctx))
    finally:
        _FakeYOLO.names = GOOD_NAMES

    check(old.read_bytes() == b"OLD-GOOD-WEIGHT",
          "⚠ 上传/发布这一步把上一版**覆盖**了（现场就是这么丢权的 ✗）：%r" % old.read_bytes()[:20])
    check("不发布" in ctx.text(),
          "类名对不上却没说「不发布」（人下次又要靠猜 ✗）：%r" % ctx.text()[-300:])
    check("person" in ctx.text() and "player" in ctx.text(),
          "日志没把两边的类名摆出来（要一眼看出「这份是 COCO」✓）：%r" % ctx.text()[-300:])


def t_live_warns_on_weight_mismatch():
    """⑤ 源码级：实时那侧"开始推理"时会校验 `player`/`mob`，缺了 ⇒ 当场告警（弹窗 + 打点）。

    ⚠ 为什么源码级就够：真跑要开 Qt + 推流 + 一份坏权重（几十 MB ✓）⇒ 用例里摆不动 ✗；
      而这条链的**要害**是"有没有这道闸" —— 缺了它，"0 框"就永远没有提示 ✓（现场正是如此 ✗）。
    """
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("weights_warn = pyqtSignal(str)" in lt,
          "实时线程没有 `weights_warn` 这个告警信号（0 框就还是没人说话 ✗）")
    check('for n in ("player", "mob") if n not in _by_name' in lt,
          "加载权重后没有校验 `player`/`mob`（现场那份 COCO 权重就是这么一路静默 ✗）")
    check("weights_warn.emit(" in lt and "weights_mismatch" in lt,
          "校验了却没上报（既没发信号、也没写 perf 打点 ✗）")
    lp = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("weights_warn.connect(" in lp and "_on_weights_warn" in lp,
          "面板没接那个信号（告警到不了人眼前 ✗）")
    check('QMessageBox.warning(self, "这份权重不是本项目的"' in lp,
          "面板收到告警没有弹出来（状态行那一行很容易被忽略 ✗）")


TESTS = (
    ("续训判据：只认 epoch>=0 + 优化器状态（正常跑完的 last.pt 不能续）",
     t_resumable_reads_epoch_and_optimizer),
    ("续不了 ⇒ 报错停下，且**连模型都不加载**（钉「不退化」✗）",
     t_not_resumable_stops_before_loading_model),
    ("真续训也带 data/project/name（退化时的保险）+ 类名对得上就正常发布",
     t_resume_also_passes_data_and_publishes_when_names_match),
    ("发布前对类名：对不上不发布，且上一版权重原封不动（现场那件事 ✗）",
     t_publish_refuses_name_mismatch_and_keeps_old_weight),
    ("实时侧：开始推理时校验 player/mob，缺了当场告警（弹窗 + 打点）",
     t_live_warns_on_weight_mismatch),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    if _TMP is not None:
        shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
