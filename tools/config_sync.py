"""双机配置一致性：哪些文件**必须一致**、怎么校验、哪些**绝不能同步**。

**为什么不做"自动同步"**（`deploy/config.py` 里已经踩过这个坑）：
    两台机器各有一份文件，改了一边另一边不会变 —— "看起来同步了、其实没有"最坑。
    而"自动双向同步"比这更糟：
      · A 机改推流参数会**悄悄改掉** B 机的收流期望（fps/端口不一致就是这么来的）；
      · 两边同时改 → 互相覆盖，谁后写谁赢，还没有痕迹；
      · 要求两台机器**同时在线**（现场经常不是）。
    所以这里的定位很小：**单向分发（以一侧为源）+ 哈希校验（在另一侧验）** ——
    把"有没有漂移"变成一个一眼能看见的结论，**不替人去改文件**。

**四组**（`--show` 原样打印，可以当文档看）：
    MUST        必须一致、**而 git 带不到**：目前只有 `remote_kbd/certs/cert.pem`
    GIT_MANAGED 由 **git** 保证一致：代码、link.yaml、候选清单（清单不重复管）
    SOFT        建议一致：现在是空的（偏好类已归 LOCAL）
    LOCAL       各机本地、**绝不同步**：deploy.json（A 机串口/屏幕区域/推流参数）、
                B 机标定出来的几何、时钟偏移、ui.yaml、各机产物……

**A 机是用 `git pull` 同步的** —— 这一点决定了上面的分工：
代码与共用配置走 git（版本化、能回溯、能下放），哈希清单只补 git **带不到**的那部分。
另外：LOCAL 里的东西会被 `.gitignore` 挡住（`selftest_config_sync` 会验），
否则 `git pull` 会把 A 机的串口/屏幕区域当成"待更新文件"覆盖掉。

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

#: **必须一致、而 git 带不到**的文件 —— 清单只管这些。
#:
#: 为什么只剩一个：代码和 `config/link.yaml`、`config/push_presets.json` 是**被 git
#: 跟踪**的（A 机 `git pull` 就同步了）—— 版本化、可回溯、能下放，那比哈希清单强得多，
#: 所以清单**不再重复管**它们。重复管的坏处很具体：忘了重跑 `--write` 就报假漂移，
#: 而人一旦被假警报咬过，就再也不信这个检查了。
#: 真正需要哈希的是"必须在两台机器上一致、却不在 git 里"的东西 —— 目前就是证书：
#: 它必须两边相同，又是各机本地生成的（当年就是因为这个才发现两边差着）。
MUST = (
    ("remote_kbd/certs/cert.pem",
     "键盘中继：B 机拿它当 cafile **校验 A 机出示的证书**"
     "（remote_kbd/kbd_client.py 的 create_default_context(cafile=...)）—— "
     "两边不是同一份，键盘就连不上/一直重连。**权威来源是 A 机**（relay 是服务端）"),
)

#: **由 git 保证一致**（不在清单里，列出来只为让人知道它们归谁管）。
GIT_MANAGED = (
    ("config/link.yaml", "双机共用的链路配置 —— 在一侧改，靠 git 同步"),
    ("config/push_presets.json", "试推候选清单 —— 同上"),
    ("tools/*.py", "握手协议、评分口径、探针解码、测量与出表……"),
    ("deploy/*.py", "A 机侧：部署台、环境自检、推流自检"),
)

#: **建议一致**（偏好类）：不一致只提示，不当作错误。
#:
#: 现在是空的：`config/ui.yaml`（字号/框色）已归入 LOCAL（各机各自的偏好，
#: 谁都不该覆盖谁）。留这个元组是为了以后真有"建议一致"的东西时有地方放。
SOFT = ()

#: **各机本地，绝不同步**：每条都写原因，省得以后有人"顺手同步一下"。
LOCAL = (
    ("config/deploy.json",
     "A 机本地：本机串口/屏幕区域/推流参数。它**派生**自 link.yaml，由环境自检比对"),
    ("remote_kbd/certs/key.pem",
     "私钥**只在 A 机**用（relay 服务端 load_cert_chain，见 remote_kbd/relay.py）；"
     "B 机从不加载它 —— 所以两边不必一致（要求同步一份私钥是没必要的）"),
    ("config/probe_calib.json", "B 机标定出来的几何（框选 / solve 的产物）"),
    ("config/ui.yaml",
     "字号与框色是**每台机器各自的偏好** —— 谁都不该覆盖谁"
     "（所以它既不进 git，也不要求两边一致）"),
    ("config/live.yaml", "B 机实时预览的偏好"),
    ("config/session.json", "B 机上次打开的会话"),
    ("config/clock_offset.txt", "B 机现测出来的时钟偏移（会漂，抄过去就是错的）"),
    ("config/decision.json", "B 机决策参数"),
    ("config/wz.yaml", "B 机本地的表数据 —— A 机根本用不到，同步过去只会造成误会"),
    ("perf_push_A.json", "A 机跑出来的产物（跑完自动发给 B，不用人拷）"),
    ("perf_push_B.json", "B 机跑出来的产物"),
)


#: 每个文件"往哪个方向拷"。**默认是"从源那一侧拷过来"，但证书反过来** ——
#: 它是 A 机（relay 服务端）出示的那一份，B 只是拿它当 cafile 去校验。
#: 不说清方向的后果实测过：A 换了新证书，B 看到"和清单不一致"，
#: 于是照提示"从源拷贝"—— 拷反了，键盘更连不上。
HINTS = {
    "remote_kbd/certs/cert.pem":
        "cert.pem 的方向**和其它文件相反**：以 **A 机**那份为准（relay 是服务端）。\n"
        "    把 A 机的 remote_kbd\\certs\\cert.pem 拷到 B 覆盖（只拷这一个，\n"
        "    key.pem 不拷）→ 重启 A 的「键盘中继」→ 在 B 跑 python -m tools.selftest_link 验证。",
}


def tracked():
    """进清单的文件（MUST + SOFT）→ ((相对路径, 为什么), ...)。"""
    return tuple(MUST) + tuple(SOFT)


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
        # 每个"必须一致"的文件，**拷贝方向可能不一样** —— 一句话说清往哪拷，
        # 免得两边互相等（或者拷反了，越弄越乱）。
        hints = [HINTS[r.split("：")[0]] for r in hard
                 if r.split("：")[0] in HINTS]
        return [{"level": "bad", "title": title,
                 "detail": "**和另一台不一致**（%s）：\n  %s\n%s"
                           % (where, "\n  ".join(hard),
                              ("\n" + "\n".join(hints)) if hints
                              else "\n把源那一侧的文件拷过来即可（别在这边手改 —— "
                                   "下次部署又会被覆盖）。")}]
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
    """打印分组 + 每条的"不一致会怎样"（当文档看）。"""
    print("共用文件一致性：分组如下（详见 tools/config_sync.py 顶部说明）\n")
    for name, items, note in (
            ("MUST        必须一致、而 git 带不到", MUST, "不一致会出明确怪现象"),
            ("SOFT        建议一致", SOFT, "不一致只提示（当前为空）"),
            ("GIT_MANAGED 由 git 保证一致（不进清单）", GIT_MANAGED,
             "在一侧改、git 同步即可，清单不重复管"),
            ("LOCAL       各机本地、绝不同步", LOCAL,
             "同步过去、或被 git 带过去，都会出错")):
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
