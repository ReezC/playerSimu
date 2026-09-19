"""键盘事件采集器：记录每个按键 down/up 事件的纳秒时间戳。

依赖: pip install pynput

用法:
    python key_capture.py --duration 30 --out human.json
    python key_capture.py --duration 20 --out inject_fix.json --delay 3

采集期间，系统里所有键盘事件（真人敲的、Pro Micro 发的）都会被记录，
用「分时段」方式区分来源：采集注入样本时不要手动敲键，采集真人样本时不要让板子发键。
"""
import argparse
import json
import time

from pynput import keyboard


def key_name(key):
    try:
        return key.char if key.char else str(key)
    except AttributeError:
        return str(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0, help="采集时长（秒）")
    ap.add_argument("--out", default="events.json", help="输出 json 路径")
    ap.add_argument("--delay", type=float, default=3.0, help="开始前的倒计时（秒），留时间切窗口")
    ap.add_argument("--key", default=None, help="只记录该字符键（如 a），过滤其他键避免污染")
    args = ap.parse_args()

    events = []

    def on_press(key):
        name = key_name(key)
        if args.key is not None and name != args.key:
            return
        events.append([time.perf_counter_ns(), "down", name])

    def on_release(key):
        events.append([time.perf_counter_ns(), "up", key_name(key)])

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    for i in range(int(args.delay), 0, -1):
        print(f"{i} 秒后开始采集...", flush=True)
        time.sleep(1)

    print(f"采集 {args.duration}s，开始！", flush=True)
    time.sleep(args.duration)
    listener.stop()

    with open(args.out, "w") as f:
        json.dump(events, f)
    downs = sum(1 for e in events if e[1] == "down")
    print(f"共 {len(events)} 个事件（down {downs} 个），已保存到 {args.out}")


if __name__ == "__main__":
    main()
