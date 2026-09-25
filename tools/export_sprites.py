"""把 Mob.wz 里的怪物 sprite 导进精灵库（WzProbe.exe 的 dump-mob 的包装）。

**为什么要有它**：精灵库（`config/wz.yaml` 的 `sprite_dir`，默认
`datasets/sprites/mob`）是标定/自动标注的模板来源 —— 缺哪只怪的目录，后面就会
报「精灵库里找不到这只怪」。导出对话框只管地图清单（`dump-map`），精灵这边一直
靠手敲 exe 命令行，路径/参数都不好记，所以收成一条：

    python -m tools.export_sprites                     # 全量（Mob.wz 里全部怪，几分钟）
    python -m tools.export_sprites --only 3000004,3000005   # 只补这几只（秒级）
    python -m tools.export_sprites --limit 20          # 只导前 20 只（试跑）
    python -m tools.export_sprites --dry-run           # 只打印要跑的命令，不动文件

**link 型的怪**（自己的 img 里只有 `info/link`，例如石人寺院III 的蝴蝶精
3000004 → 3000001）必须用**新版** WzProbe：老版会一只都导不出来。脚本会检查
exe 在哪（`config/wz.yaml` 的 `probe_exe`，否则按 core/wzexport 的候选路径探测），
并把 exe 的输出原样转出来。
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import wzexport


def _survey(root):
    """精灵库现状 → (目录数, 一帧都没有的目录数, 只有 fly 没有 stand 的目录数)。"""
    if not root.is_dir():
        return 0, 0, 0
    dirs = [d for d in root.iterdir() if d.is_dir()]
    empty = sum(1 for d in dirs if not wzexport.mob_action_frames(d)[1])
    fly_only = sum(1 for d in dirs
                   if not list(d.glob("stand_*.png")) and list(d.glob("fly_*.png")))
    return len(dirs), empty, fly_only


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 Mob.wz 的怪物 sprite 到精灵库")
    ap.add_argument("--only", default=None, metavar="ID,ID",
                    help="只导这些怪 id（逗号分隔）—— 补几只时用，秒级完成")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多导几只（0 = 全部）；试跑用")
    ap.add_argument("--exe", default=None, help="WzProbe.exe 路径（默认读配置/自动探测）")
    ap.add_argument("--wz", default=None, help="WZ 目录（默认读配置）")
    ap.add_argument("--out", default=None, help="精灵库目录（默认读配置）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不真的导出")
    args = ap.parse_args()

    cfg = wzexport.load_cfg()
    exe = wzexport.find_probe_exe(args.exe or cfg.get("probe_exe") or "")
    wz = args.wz or cfg.get("wz_dir") or ""
    out = Path(args.out) if args.out else wzexport.sprite_dir_path()

    print("WzProbe : %s" % (exe or "（没找到）"))
    print("WZ 目录 : %s" % (wz or "（没配置）"))
    print("精灵库  : %s" % out)

    if not exe or not Path(exe).exists():
        print()
        print("找不到 WzProbe.exe。先编译：")
        print("  cd E:\\MyPrograms\\RippleRogue\\MapleNecrocer")
        print("  dotnet build WzProbe/WzProbe.csproj -c Release")
        print("（或在 config/wz.yaml 的 probe_exe 里填绝对路径）")
        return 2
    if not wz or not Path(wz).is_dir():
        print()
        print("WZ 目录不对（config/wz.yaml 的 wz_dir）。")
        return 2
    if not (Path(wz) / "Mob.wz").exists():
        print()
        print("这个 WZ 目录里没有 Mob.wz：%s" % wz)
        return 2

    cmd = [exe, "dump-mob", str(wz), str(out), str(int(args.limit))]
    if args.only:
        cmd += ["--only", args.only]

    print()
    print("命令: %s" % " ".join('"%s"' % c if " " in c else c for c in cmd[1:]))
    if args.dry_run:
        return 0

    before = _survey(out)
    print("导出前：目录 %d 个（其中一帧都没有 %d 个，只有 fly 没 stand %d 个）"
          % before)
    print()

    # 逐行转出 exe 的输出 —— 全量要几分钟，不能让它看起来卡死
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print("  " + line, flush=True)
    except KeyboardInterrupt:
        proc.terminate()
        print("\n已中断")
    code = proc.wait()

    after = _survey(out)
    print()
    print("导出后：目录 %d 个（其中一帧都没有 %d 个，只有 fly 没 stand %d 个）"
          % after)
    print("  净增目录 %d 个；补上了 %d 个原本一帧都没有的"
          % (after[0] - before[0], before[1] - after[1]))
    if code != 0:
        print("WzProbe 退出码 %d（有怪物导出失败，上面的 FAIL 行里有原因）" % code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
