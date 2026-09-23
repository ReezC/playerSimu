"""同步 Pro Micro 固件到 Arduino 目录（项目版 → 烧录副本）。

项目里的 remote_kbd/pro_micro/pro_micro.ino 是唯一源，而 Arduino IDE 编译的是
Documents/Arduino/<sketch>/<sketch>.ino。两份不同步就会烧进旧固件（少命令、
行为不一致 —— 比如鼠标命令没烧进去，摇杆点了没反应）。

用法：
    python tools/sync_firmware.py                 # 同步到默认 Arduino 目录
    python tools/sync_firmware.py --check         # 只对比，不写入（返回码 2 = 不一致）
    python tools/sync_firmware.py --dest <路径>   # 指定目标：.ino 文件或其所在目录
"""

import argparse
import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "remote_kbd" / "pro_micro" / "pro_micro.ino"
DEFAULT_DEST = Path.home() / "Documents" / "Arduino" / "1" / "1.ino"


def resolve_dest(dest):
    """目标可以是 .ino 文件，或它所在的目录（目录名 = sketch 名，文件名同名）。"""
    if dest is None:
        return DEFAULT_DEST
    p = Path(dest)
    if p.is_dir():
        return p / (p.name + ".ino")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None, help="目标 .ino 文件或其所在目录")
    ap.add_argument("--check", action="store_true", help="只对比，不写入")
    args = ap.parse_args()

    if not SRC.exists():
        print("找不到源固件:", SRC)
        return 1

    dest = resolve_dest(args.dest)
    print("源:  ", SRC)
    print("目标:", dest)

    if not dest.parent.exists():
        print("\n目标目录不存在 —— 检查 --dest，或 Arduino 的 sketchbook 路径设置")
        return 1

    same = dest.exists() and filecmp.cmp(str(SRC), str(dest), shallow=False)
    if same:
        print("\n已是最新，无需同步。")
        return 0

    if args.check:
        print("\n两份不一致（--check 模式，未写入）。")
        return 2

    if dest.exists():
        bak = dest.with_suffix(dest.suffix + ".bak")
        shutil.copy(dest, bak)
        print("\n旧版已备份 ->", bak)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC, dest)
    print("已同步 ->", dest)
    print("\n下一步：Arduino IDE 打开该文件 → 上传"
          "（上传瞬间双击 RESET 进 bootloader）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
