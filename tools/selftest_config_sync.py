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

    # ★ 光有 .gitignore **不算数**：它对**已经被跟踪**的文件无效。
    # 实测踩过（2026-09-25）：规则写了，但那 11 个本机文件仍被跟踪 —— 于是
    # `git pull` 一路要求"清理"，而且被误以为"已经移出版本控制了"（连我都被误导）。
    # 所以这里直接用 git 查**真实的跟踪状态**（不是查文件里有没有那条规则）。
    import subprocess

    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except Exception:                                # noqa: BLE001
        print("      （这台机器没有 git，跳过跟踪状态检查）")
    else:
        tracked_bad = []
        # LOCAL 那批 + 证书（cert.pem 在 MUST 里、key.pem 在 LOCAL 里 —— 它们
        # **都不该被 git 带**，所以这里一起查）
        for rel in [r for r, _w in cs.LOCAL] + ["remote_kbd/certs/cert.pem"]:
            if not (cs.ROOT / rel).exists():
                continue
            r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                               cwd=str(cs.ROOT), capture_output=True, text=True)
            if r.returncode == 0:
                tracked_bad.append(rel)
        check(not tracked_bad,
              "这些本机文件**仍被 git 跟踪**（.gitignore 对已跟踪文件无效）——\n"
              "        在源那一侧跑一次：\n"
              "          git rm --cached %s\n"
              "        然后 git add -A && git commit && git push；"
              "A 机备份好本机文件再 git pull。"
              % " ".join(tracked_bad))


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


def t_local_pair_is_checked_first():
    """跨机比较之前，先看**本机这一对证书本身**配不配对。

    实测踩过（2026-09-25）：A 上 `cert.pem`/`key.pem` 被手工换混了（一次新的、一次旧的）
    → relay 起不来（`KEY_VALUES_MISMATCH`），拷到 B 的那份也用不了，而两边清单都"一致"。
    等于拿一把坏尺子比长度：所以本机配对要排在最前面。
    """
    import shutil

    from remote_kbd import gen_cert

    root = _fake_root()
    d1 = Path(tempfile.mkdtemp(prefix="c1_"))
    d2 = Path(tempfile.mkdtemp(prefix="c2_"))
    try:
        c1, _k1 = gen_cert.generate(d1)
        _c2, k2 = gen_cert.generate(d2)
        certs = root / "remote_kbd" / "certs"
        certs.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(c1), str(certs / "cert.pem"))      # 第一对的证书
        shutil.copyfile(str(k2), str(certs / "key.pem"))       # 第二对的私钥 → 必然不配对
        cs.write(root)
        rows = cs.check(root)
        check(rows[0]["level"] == "warn" and "配不上" in rows[0]["title"],
              "混了对的证书没被指出来：%s" % rows)
        check("生成证书" in rows[0]["detail"],
              "没说怎么修（应当指向一键生成）：%s" % rows[0]["detail"])
        # 只在"跑键盘中继那台"才是错（部署台的 check_certs 报 bad）；
        # 这里在两边都跑，所以只能是提示 —— 否则控制机上天天误报
        check("控制机只用 cert.pem" in rows[0]["detail"],
              "没说明只有 A 机需要它配对：%s" % rows[0]["detail"])
    finally:
        for d in (root, d1, d2):
            shutil.rmtree(str(d), ignore_errors=True)


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
    ("先看本机这对证书自己配不配对", t_local_pair_is_checked_first),
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
