"""从**新版（Steam/GMS 那种分包布局）**客户端导宠物 → 一个输出目录 ✓。

    python -X utf8 tools/export_pets_steam.py 5002828,5002829 --out %TEMP%\\pets_out

为什么要单独一个工具（用户 2026-10-04 ✓ 原话："单独从
`E:\\SteamLibrary\\steamapps\\common\\MapleStory\\Data\\` 这里的 Wz 导出一下这 4 个宠物"）：
新版客户端**不是**"一个 `Item.wz` 装全部"的老布局 ✗，WzProbe 直接指过去一个都读不到
（实测报 `没找到宠物分支` ✓）⇒ 必须先**拼一份兼容布局**（下面那三份 copy ✓）：

    Data/Item/Pet/Pet_000.wz        →  <tmp>/Item.wz        （`/Pet` 节点**自己**就是一个 wz ✓
                                                              根下面直接挂 `<id>.img` ✓）
    Data/String/String_000.wz       →  <tmp>/String.wz      （`String.wz` 只是 63 字节空壳 ✗
                                                              真名字在 `String_000.wz` ✓）
    Data/Item/Pet/_Canvas/_Canvas_000.wz → <tmp>/_Canvas.wz （⭐ 帧上**没有像素** ✗ 只有
                                                              `_outlink = Item/Pet/_Canvas/…` ✓
                                                              真像素全在这份里 ✓）

⚠ 依赖 `WzProbe`（外部仓库 ✓）的两处改动（见 SKILL 约定 **217** ✓）：
  ① `FindPetGroups` 认"**根自己就是 `/Pet` 节点**"✓（老判据只认有个叫 `Pet` 的分组 ✗）；
  ② 取像素前**跟随 `_outlink`/`_inlink`** ✓（`wzDir` 里有 `_Canvas.wz` 就自动载入 ✓）。
  没有 ② 的话**帧数全对、图全是 1×1 空图** ✗（实测踩过 ✓）⇒ 导完务必查"不透明像素 > 0"✓。

⚠ 本工具**只导出**（写到你给的 `--out` ✓）：不碰 `datasets/` ✗、不改 `pets.json` ✗
  —— 要并进仓库，照 SKILL 217 那步手工拷 + 追加清单 ✓（免得把老客户端的 50 只覆盖掉 ✗）。
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

#: 默认的 Steam 客户端数据目录（用户 2026-10-04 给的 ✓）
DEFAULT_DATA = r"E:\SteamLibrary\steamapps\common\MapleStory\Data"
#: 默认的 WzProbe（外部仓库 ✓ 见 `config/wz.yaml` 的 `probe_exe` 是同一个 ✓）
DEFAULT_EXE = r"E:\MyPrograms\RippleRogue\MapleNecrocer\WzProbe\bin\Release\net8.0\WzProbe.exe"


def main(argv=None):
    ap = argparse.ArgumentParser(description="从新版（分包）客户端导宠物")
    ap.add_argument("ids", help="宠物 id，逗号分隔（例：5002828,5002829）")
    ap.add_argument("--data", default=DEFAULT_DATA, help="客户端 Data 目录")
    ap.add_argument("--exe", default=DEFAULT_EXE, help="WzProbe.exe")
    ap.add_argument("--out", default="", help="输出目录（默认一个临时目录 ✓）")
    ap.add_argument("--probe", action="store_true", help="把结构一并打出来（排查用 ✓）")
    a = ap.parse_args(argv)

    data = pathlib.Path(a.data)
    out = pathlib.Path(a.out) if a.out else pathlib.Path(tempfile.mkdtemp(prefix="pets_steam_"))
    if not data.is_dir():
        print("✗ 没有这个 Data 目录:", data)
        return 2
    if not pathlib.Path(a.exe).is_file():
        print("✗ 找不到 WzProbe.exe:", a.exe)
        return 2

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="wz_steam_"))
    plan = ((data / "Item" / "Pet" / "Pet_000.wz", tmp / "Item.wz"),
            (data / "String" / "String_000.wz", tmp / "String.wz"),
            (data / "Item" / "Pet" / "_Canvas" / "_Canvas_000.wz", tmp / "_Canvas.wz"))
    for src, dst in plan:
        if not src.is_file():
            print("✗ 这份客户端里没有:", src)
            return 2
        shutil.copy2(src, dst)
        print("  %-46s → %-13s %.1f MB" % (src.name, dst.name, dst.stat().st_size / 1e6))

    cmd = [a.exe, "dump-pets", str(tmp), str(out), "--only", a.ids]
    if a.probe:
        cmd.append("--probe")
    print("跑:", " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
    for l in ((p.stdout or "") + (p.stderr or "")).splitlines():
        print("   " + l.rstrip())
    print("rc =", p.returncode)
    print("产物目录:", out)
    print("⚠ 记得查一眼**像素**（帧数对但全 1×1 = 没跟随 `_outlink` ✗ 见文件头说明 ✓）")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
