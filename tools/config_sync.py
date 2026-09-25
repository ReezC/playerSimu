"""双机配置一致性：哪些文件**必须一致**、怎么校验、哪些**绝不能同步**。

**为什么不做"自动同步"**（`deploy/config.py` 里已经踩过这个坑）：
    两台机器各有一份文件，改了一边另一边不会变 —— "看起来同步了、其实没有"最坑。
    而"自动双向同步"比这更糟：
      · A 机改推流参数会**悄悄改掉** B 机的收流期望（fps/端口不一致就是这么来的）；
      · 两边同时改 → 互相覆盖，谁后写谁赢，还没有痕迹；
      · 要求两台机器**同时在线**（现场经常不是）。
    所以这里的定位很小：**单向分发（以一侧为源）+ 哈希校验（在另一侧验）** ——
    把"有没有漂移"变成一个一眼能看见的结论，**不替人去改文件**。

**三档**（`--show` 原样打印，可以当文档看）：
    MUST    必须一致：link.yaml / 候选清单 / 证书对 / **接线相关的那几个 .py**
    SOFT    建议一致：ui.yaml（字号与框色是人的偏好，不一致只提示）
    LOCAL   各机本地、**绝不同步**：deploy.json（A 机物理布局 + 本机参数）、
            B 机标定出来的几何、时钟偏移、各机产物……

**用它就三步**：
    1. 在"源"那一侧（通常是改代码/配置的那台，比如 B 机）：
           python -m tools.config_sync --write
       它把共用文件的 sha256 写进 `config/sync_manifest.json`（随仓库一起拷过去）；
    2. 照常把仓库拷到另一台机器（现有的部署做法不变）；
    3. 在另一台机器上：部署台的**环境自检**会多一行"配置与部署清单"，
       或者手动 `python -m tools.config_sync --check` —— 不一致就明确列出是哪个文件。

**为什么"部署台里那不已经在比对了吗"还不够**：`deploy/selfcheck.py` 的
`check_link` 比的是**同一台机器上** link.yaml ↔ deploy.json（本机自洽）；
它证明不了"A 机的 link.yaml 和 B 机的 link.yaml 是同一份" ——
今天 fps/端口那次不一致，正是这个缝里漏出来的。清单把这条补上。
"""

import hashlib
import json
import socket
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: **必须一致**的文件（每条都写清"不一致会出现什么现象"，省得有人觉得无关紧要）。
MUST = (
    ("config/link.yaml",
     "收流地址/端口/分辨率/帧率/探针换算 —— 不一致就是「流发出来了、B 机没画面」"),
    ("config/push_presets.json",
     "试推的候选清单 —— 不一致两边跑的不是同一份（有握手后不致命，但表会串味）"),
    ("remote_kbd/certs/cert.pem", "键盘中继的证书是**配对**的，不一致就连不上"),
    ("remote_kbd/certs/key.pem", "同上（私钥与证书必须成对）"),
)

#: 接线相关代码：不一致会出现"一边在等、一边没发"这种最难查的现象。
#: 只列**不一致就会出怪现象**的那几个，不是全仓库（全仓库比对留给 git）。
MUST_CODE = (
    ("tools/sweep_link.py", "握手协议本体 —— 不一致就永远对不上（一方听不懂公告）"),
    ("tools/push_presets.py", "记账字段与评分口径 —— 不一致会合并出空的表"),
    ("tools/probe_codec.py", "探针几何与解码 —— 两边口径必须是同一份"),
    ("tools/stream_sweep.py", "B 机侧：测量、收报告、出表"),
    ("deploy/push_sweep.py", "A 机侧：公告、发报告、等结论"),
    ("deploy/config.py", "A 机配置的键集合 —— 缺键会被 read() 静默丢掉"),
)

#: **建议一致**（偏好类）：不一致只提示，不当作错误。
SOFT = (
    ("config/ui.yaml",
     "字号与框色是人的偏好 —— 不一致只是「两个界面长得不一样」，不影响功能"),
)

#: **各机本地，绝不同步**：每条都写原因，省得以后有人"顺手同步一下"。
LOCAL = (
    ("config/deploy.json",
     "A 机本地：本机串口/屏幕区域/推流参数。它**派生**自 link.yaml，由环境自检比对"),
    ("config/probe_calib.json", "B 机标定出来的几何（框选 / solve 的产物）"),
    ("config/live.yaml", "B 机实时预览的偏好"),
    ("config/session.json", "B 机上次打开的会话"),
    ("config/clock_offset.txt", "B 机现测出来的时钟偏移（会漂，抄过去就是错的）"),
    ("config/decision.json", "B 机决策参数"),
    ("config/wz.yaml", "B 机本地的表数据 —— A 机根本用不到，同步过去只会造成误会"),
    ("perf_push_A.json", "A 机跑出来的产物（跑完自动发给 B，不用人拷）"),
    ("perf_push_B.json", "B 机跑出来的产物"),
)


def tracked():
    """进清单的文件（MUST + MUST_CODE + SOFT）→ ((相对路径, 为什么), ...)。"""
    return tuple(MUST) + tuple(MUST_CODE) + tuple(SOFT)


def _is_soft(rel):
    return rel in tuple(r for r, _w in SOFT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def manifest_path(root=ROOT):
    return Path(root) / "config" / "sync_manifest.json"


def write(root=ROOT, manifest=None):
    """在**源那一侧**生成清单 → (写好了吗, 说明文字)。"""
    root = Path(root)
    man = Path(manifest) if manifest else manifest_path(root)
    files, missing = {}, []
    for rel, _why in tracked():
        p = root / rel
        if p.exists():
            files[rel] = sha256(p)
        else:
            missing.append(rel)
    data = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(), "files": files}
    try:
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    except Exception as e:                       # noqa: BLE001
        return False, "写不了 %s：%s" % (man, e)
    return True, ("已写 %s（%d 个文件%s）"
                  % (man.name, len(files),
                     "；**缺 %d 个**：%s" % (len(missing), "、".join(missing))
                     if missing else ""))


def check(root=ROOT, manifest=None):
    """比对 → 与 `deploy/selfcheck.py` 同形的行：[{level, title, detail}]。"""
    root = Path(root)
    man = Path(manifest) if manifest else manifest_path(root)
    title = "配置与部署清单"
    if not man.exists():
        return [{"level": "warn", "title": title,
                 "detail": "没有 %s —— 还没在源那一侧跑过 "
                           "`python -m tools.config_sync --write`。\n"
                           "它回答的是「两台机器的共用文件是不是同一份」"
                           "（本机自洽那部分由「与 link.yaml 一致」那几项管）。"
                           % man.name}]
    try:
        data = json.loads(man.read_text(encoding="utf-8"))
        want = dict(data.get("files") or {})
    except Exception as e:                       # noqa: BLE001
        return [{"level": "bad", "title": title, "detail": "清单读不了：%s" % e}]

    hard, soft, missing = [], [], []
    for rel, _why in tracked():
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        got = sha256(p)
        if rel not in want:
            (soft if _is_soft(rel) else hard).append("%s：清单里没有这一项" % rel)
        elif got != want[rel]:
            (soft if _is_soft(rel) else hard).append(
                "%s：本机 %s… vs 清单 %s…" % (rel, got[:8], want[rel][:8]))
    if missing:
        hard.extend("%s：本机**找不到这个文件**" % m for m in missing)

    where = "清单写于 %s（%s）" % (data.get("when", "?"), data.get("host", "?"))
    if hard:
        return [{"level": "bad", "title": title,
                 "detail": "**和另一台不一致**（%s）：\n  %s\n"
                           "把源那一侧的文件拷过来即可（别在这边手改 —— "
                           "下次部署又会被覆盖）。" % (where, "\n  ".join(hard))}]
    if soft:
        return [{"level": "warn", "title": title,
                 "detail": "共用文件一致；只有偏好类不同（%s）：\n  %s"
                           % (where, "\n  ".join(soft))}]
    return [{"level": "ok", "title": title,
             "detail": "%d 个共用文件都和清单一致（%s）" % (len(want), where)}]


def _msgbox(title, text, flags=0x30):
    """Windows 弹窗（**不依赖 Qt** —— 它要在 GUI 起来之前用）。非 Windows 退回打印。"""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, str(text), str(title), flags)
    except Exception:                            # noqa: BLE001
        print("%s\n%s" % (title, text))


def preflight(root=ROOT, manifest=None, msgbox=False):
    """启动器用的"一句话预检" → 退出码（0 = 一致，3 = **有漂移**）。

    **为什么单独一个入口**：`被控机部署台.bat` 用 `pythonw` 起 GUI（**没有控制台**），
    漂移只 `print` 就等于没提示 —— 它会一直隐形，直到某个功能怪怪地退回老路子
    （比如握手静默退回"按清单顺序走"）。所以这里用 Windows 弹窗说一声，
    再用退出码让 .bat 记一笔。**永不挡人**：弹完照常启动。
    """
    rows = check(root, manifest)
    level = rows[0]["level"] if rows else "warn"
    detail = rows[0]["detail"] if rows else ""
    first = detail.splitlines()[0] if detail else ""
    print("[config_sync] %s：%s" % (level.upper(), first))
    if level == "bad" and msgbox:
        _msgbox("部署台：配置/代码与部署清单不一致",
                detail + "\n\n"
                "（部署台照常启动；这条也会出现在界面右上角的环境自检里。\n"
                "处理：把源那一侧的同名文件拷过来，别在这边手改。）")
    return 3 if level == "bad" else 0


def show():
    """打印三档分类 + 每条"不一致会怎样"（当文档看）。"""
    print("共用文件一致性：三档（详见 tools/config_sync.py 顶部说明）\n")
    for name, items, note in (
            ("MUST  必须一致", tuple(MUST) + tuple(MUST_CODE),
             "不一致会出明确怪现象"),
            ("SOFT  建议一致", SOFT, "不一致只提示（偏好类）"),
            ("LOCAL 各机本地，绝不同步", LOCAL, "同步过去反而会出错")):
        print("%s  —— %s" % (name, note))
        for rel, why in items:
            print("    %-34s %s" % (rel, why))
        print()


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="在源那一侧生成/更新 config/sync_manifest.json")
    ap.add_argument("--check", action="store_true", help="比对本机与清单")
    ap.add_argument("--show", action="store_true", help="打印三档分类（当文档看）")
    ap.add_argument("--preflight", action="store_true",
                    help="启动器用：一行结论 +（可选）弹窗；退出码 3 = 有漂移")
    ap.add_argument("--msgbox", action="store_true",
                    help="漂移时弹 Windows 弹窗（GUI 用 pythonw 起、没有控制台时用）")
    ap.add_argument("--root", default=None, help="仓库根（默认自动定位）")
    args = ap.parse_args()
    root = Path(args.root) if args.root else ROOT

    if args.show or not (args.write or args.check or args.preflight):
        show()
    if args.write:
        ok, msg = write(root)
        print(("✓ " if ok else "✗ ") + msg)
    if args.check:
        for r in check(root):
            print("[%s] %s\n      %s" % (r["level"].upper(), r["title"],
                                         r["detail"].replace("\n", "\n      ")))
    if args.preflight:
        return preflight(root, None, args.msgbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
