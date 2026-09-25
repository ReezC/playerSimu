"""双机配置一致性自检：三档分类不重不漏、改一个字节就能发现、坏清单不抛。

为什么值得钉：这东西的价值全在"改了一个字节就报出来"。要是它自己宽松
（清单缺项也放过、或者比的是"文件在不在"而不是内容），那它就是摆设 ——
而它要防的正是"看起来同步了、其实没有"。
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import config_sync as cs                                 # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _fake_root():
    """造一个假仓库：清单里登记的每个文件都给一份内容（+ config/ 目录，放清单用）。"""
    root = Path(tempfile.mkdtemp(prefix="cfgsync_"))
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rel, _why in cs.tracked():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("内容 %s\n" % rel, encoding="utf-8")
    return root


def _first_must():
    """清单里第一个"必须一致"的文件（用例改它来制造漂移）。"""
    return cs.MUST[0][0]


def t_three_tiers_are_clean():
    """三档分类**不重不漏**，且登记的文件在真仓库里都存在、都写了"不一致会怎样"。

    为什么会漏：这是一张**手维护的分类表** —— 加了新的共用文件忘了登记，
    校验就永远不会提它，而漂移最爱藏在那儿。LOCAL 混进 MUST 更危险：
    那会把"本机该有的东西"同步成 A 机的布局（串口、屏幕区域）。
    """
    must = [r for r, _w in cs.MUST]
    soft = [r for r, _w in cs.SOFT]
    local = [r for r, _w in cs.LOCAL]
    check(not (set(must) & set(soft)), "MUST 与 SOFT 重叠：%s" % (set(must) & set(soft),))
    check(not (set(must) & set(local)), "MUST 与 LOCAL 重叠（会把本机文件同步掉！）")
    check(not (set(soft) & set(local)), "SOFT 与 LOCAL 重叠")
    for rel, why in cs.tracked():
        check((cs.ROOT / rel).exists(), "清单里登记的 %s 在仓库里不存在" % rel)
        check(len(why) > 6, "%s 没写「不一致会怎样」（以后没人知道它重要）" % rel)
    for rel, why in cs.LOCAL:
        check(len(why) > 6, "LOCAL 的 %s 没写原因（以后有人会顺手同步它）" % rel)

    # ★ A 机是 `git pull` 同步的 —— 所以**各机本地的文件必须被 .gitignore 挡住**，
    # 否则 pull 会把 A 机的串口/屏幕区域当"待更新文件"覆盖掉（真发生过：
    # config/deploy.json、config/ui.yaml、连 remote_kbd/certs/key.pem 都被跟踪着）。
    gi = (cs.ROOT / ".gitignore").read_text(encoding="utf-8")
    for rel, _w in cs.LOCAL:
        check(rel in gi, "%s 是各机本地的，却没被 .gitignore 挡住" % rel)
    check("remote_kbd/certs/" in gi, "证书目录没被 .gitignore 挡住（私钥不该进 git）")
    # 反过来：**该靠 git 带过去的共用文件绝不能被忽略**（否则 A 机永远拉不到）
    for rel in ("config/link.yaml", "config/push_presets.json",
                "config/sync_manifest.json"):
        check(rel not in gi, "%s 是共用文件，被 .gitignore 挡住 = A 机拉不到" % rel)


def t_detects_one_byte_change():
    """改一个字节就要报不一致；偏好类只提示；没改则报一致。"""
    root = _fake_root()
    try:
        ok, msg = cs.write(root)
        check(ok, "写清单失败：%s" % msg)
        rows = cs.check(root)
        check([r["level"] for r in rows] == ["ok"],
              "刚写完清单就报不一致：%s" % rows)

        rel = _first_must()
        p = root / rel
        p.write_text(p.read_text(encoding="utf-8") + "x", encoding="utf-8")
        rows = cs.check(root)
        check(rows[0]["level"] == "bad", "改了 %s 却没报不一致：%s" % (rel, rows))
        check(rel in rows[0]["detail"], "没说是哪个文件：%s" % rows[0]["detail"])

        # **本机文件不参与检查**：ui.yaml 是每台机器各自的偏好，两台不一样是正常的，
        # 报它只会制造假警报（人被假警报咬过就再也不信这个检查了）。
        p.write_text("内容 %s\n" % rel, encoding="utf-8")                 # 还原
        p2 = root / "config" / "ui.yaml"                                 # 本机文件（不在清单里）
        p2.write_text("字体：16\n", encoding="utf-8")
        rows = cs.check(root)
        check(rows[0]["level"] == "ok",
              "本机文件（ui.yaml）改动了不该触发漂移：%s" % rows)
    finally:
        shutil.rmtree(str(root), ignore_errors=True)


def t_missing_or_broken_is_loud_not_crash():
    """没有清单 / 清单坏了 / 文件缺了：都要说话，不许抛，也不许当没事。"""
    root = _fake_root()
    try:
        rows = cs.check(root)                     # 还没写过清单
        check(rows[0]["level"] == "warn" and "sync_manifest" in rows[0]["detail"],
              "没有清单时的提示不对：%s" % rows)

        (root / "config" / "sync_manifest.json").write_text("{坏了", encoding="utf-8")
        rows = cs.check(root)
        check(rows[0]["level"] == "bad", "坏清单应当是 bad：%s" % rows)

        cs.write(root)
        (root / _first_must()).unlink()
        rows = cs.check(root)
        check(rows[0]["level"] == "bad" and "找不到" in rows[0]["detail"],
              "文件缺了没报出来：%s" % rows)
    finally:
        shutil.rmtree(str(root), ignore_errors=True)


def t_preflight_exit_codes():
    """启动器（`被控机部署台.bat`）靠**退出码**分流：一致 → 0，漂移 → 3。

    为什么钉它：.bat 里写的是 `if errorlevel 3 ( ... >> deploy_sync.log )`。
    退出码一变，那条分支就永远不触发，而且**没人会发现** —— 弹窗照弹（也是它弹的）、
    日志却没有，下次只能靠人回忆"好像弹过一次"。
    """
    root = _fake_root()
    try:
        cs.write(root)
        check(cs.preflight(root) == 0, "一致时该返回 0")
        rel = _first_must()
        p = root / rel
        p.write_text(p.read_text(encoding="utf-8") + "x", encoding="utf-8")
        check(cs.preflight(root) == 3, "共用文件漂移时该返回 3")

        # 本机文件（ui.yaml）不一致**不该**算漂移
        p.write_text("内容 %s\n" % rel, encoding="utf-8")
        (root / "config" / "ui.yaml").write_text("字体：16\n", encoding="utf-8")
        check(cs.preflight(root) == 0, "本机文件不一致不该返回 3")
    finally:
        shutil.rmtree(str(root), ignore_errors=True)


CASES = (
    ("三档分类不重不漏、登记的文件都在", t_three_tiers_are_clean),
    ("改一个字节就报不一致（偏好类只提示）", t_detects_one_byte_change),
    ("缺清单/坏清单/缺文件都说话但不抛", t_missing_or_broken_is_loud_not_crash),
    ("启动器预检的退出码（一致0 / 漂移3）", t_preflight_exit_codes),
)


def main():
    bad = 0
    for name, fn in CASES:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                    # noqa: BLE001
            bad += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
