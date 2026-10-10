"""查「本仓库上一次遗留的 python 进程」—— **装机/长任务留下的看门狗与卡死的 pip**。

为什么要有它（2026-10-10 现场 ✓ 用户原话："为什么会有遗留 python？杀掉并严查"）：
    早先装 `visual_tracking_sdk` 那次，为了盯一个很长的 `pip install`，起了一个**临时看门狗**
    （`_tmp_install_watch.py` ✗）；脚本后来被删了，**进程却一直在** ✓ —— 实测它空转了
    **2163 CPU 秒**（7 小时 ✓），下面还挂着**两个卡死的 pip**（CPU 0 / 内存 0 ✓）。
    人不可能天天开任务管理器数进程 ✗ ⇒ 给一条命令：**列出来 + 说清它是什么 +（可选）清掉** ✓。

判据（保守 ✓ 只认"肯定是我们留下的"这几种，**绝不误杀工作台/编辑器** ✗）：
    · 命令行里有 `_tmp_` / `install_watch` ⇒ 吃安装看门狗 ✓；
    · 命令行里有 `pip install` 而且**已经很老**（`--min-age` 秒 ✓ 默认 900）⇒ 卡住的安装 ✓；
    · `python -m gui.app`（工作台）**永不列入** ✗ —— 那是用户正在用的 ✓。

跑法：
    python -m tools.stale_check              # 只看（默认）
    python -m tools.stale_check --kill       # 看 + 杀（杀之前逐条打印 ✓ 留痕）
    python -m tools.stale_check --min-age 300
"""
import argparse
import subprocess
import sys
import time

#: 只认这几种"肯定是我们留下的"（**白名单式** ✓ 绝不按"python 进程"一把抓 ✗）
PATTERNS = (("_tmp_", "临时脚本（多半是安装看门狗 ✓）"),
            ("install_watch", "安装看门狗 ✓"),
            ("-m pip install", "pip 安装（卡住的那种 ✓）"),
            ("pip install -r", "pip 安装（卡住的那种 ✓）"))

#: ⛔ **永不列入**（用户正在用的东西 ✓）
NEVER = ("-m gui.app", "gui/app.py", "selftest", "stale_check")


def _query():
    """`Get-CimInstance` 拉进程（含命令行/父进程/启动时刻/CPU 秒）→ list[dict]。"""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
          "Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate,UserModeTime,"
          "WorkingSetSize | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60)
    except Exception as e:                          # noqa: BLE001
        print("查不动进程：%s" % e)
        return []
    txt = (out.stdout or "").strip()
    if not txt:
        return []
    import json
    try:
        data = json.loads(txt)
    except Exception:                               # noqa: BLE001
        return []
    return data if isinstance(data, list) else [data]


def _age_s(rec):
    """这个进程活了多久（秒）；拿不到时间 ⇒ 0（保守：不杀 ✓）。"""
    raw = str(rec.get("CreationDate") or "")
    # CIM 给的是 "/Date(1759...)/" 这种 ✓
    import re
    m = re.search(r"(\d{10,})", raw)
    if not m:
        return 0.0
    try:
        t = float(m.group(1)) / 1000.0
    except ValueError:
        return 0.0
    if t > 1e11:                                    # 纳秒
        t /= 1e6
    return max(0.0, time.time() - t)


def find(min_age_s=900.0):
    """→ [(pid, 为什么算遗留, 命令行, 活了多久秒, CPU秒)]（**只读** ✓）。"""
    out = []
    for rec in _query():
        try:
            cmd = str(rec.get("CommandLine") or "")
            pid = int(rec.get("ProcessId") or 0)
        except (TypeError, ValueError):
            continue
        if not cmd or not pid:
            continue
        if any(n in cmd for n in NEVER):
            continue                                # ⛔ 工作台/自检：永不列入 ✓
        why = None
        for pat, desc in PATTERNS:
            if pat in cmd:
                why = desc
                break
        if why is None:
            continue
        age = _age_s(rec)
        if "pip install" in cmd and age < float(min_age_s):
            continue                                # 刚开始装的别动 ✓（可能是人正在装的 ✓）
        cpu = float(rec.get("UserModeTime") or 0) / 1e7
        out.append((pid, why, cmd, age, cpu))
    out.sort(key=lambda r: -r[4])                   # 按烧掉的 CPU 排 ✓（谁最可疑一目了然 ✓）
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill", action="store_true", help="看 + 杀掉（逐条打印 ✓）")
    ap.add_argument("--min-age", type=float, default=900.0,
                    help="`pip install` 至少活了这么久才算遗留（秒 ✓ 默认 900）")
    a = ap.parse_args()

    rows = find(a.min_age)
    print("遗留 python 进程：%d 个" % len(rows))
    for pid, why, cmd, age, cpu in rows:
        print("  PID %-7d 活了 %6.0f 秒  烧了 %7.1f CPU 秒  %s" % (pid, age, cpu, why))
        print("      %s" % cmd[:160])
    if not rows:
        print("  （干净 ✓ —— 工作台 `-m gui.app` 永不列入 ✓）")
        return 0
    if not a.kill:
        print("\n只查看模式（加 `--kill` 才动手 ✓）")
        return 0
    for pid, why, cmd, age, cpu in rows:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=30)
            print("  已杀 PID %d（%s ✓）" % (pid, why))
        except Exception as e:                      # noqa: BLE001
            print("  杀 PID %d 失败：%s" % (pid, e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
