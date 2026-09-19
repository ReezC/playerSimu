"""向 Pro Micro 串口发注入命令（配合固件 key_injector）。

依赖: pip install pyserial

用法:
    python send_cmd.py COM5 "FIX a 100 100"      # 固定 100ms 间隔按 a 键 100 次
    python send_cmd.py COM5 "RND a 60 140 100"   # 60~140ms 均匀随机间隔按 a 键 100 次

注意: 打开串口会触发 Pro Micro 复位（DTR），脚本会等待 2 秒再发命令。
"""
import sys
import time

import serial


def main():
    if len(sys.argv) < 3:
        print('用法: python send_cmd.py <COM端口> "<命令>"')
        return
    port = sys.argv[1]
    cmd = " ".join(sys.argv[2:])

    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2.0)  # 等 Pro Micro 复位完成
    ser.write((cmd + "\n").encode())
    ser.flush()

    # 等固件回 DONE
    deadline = time.time() + 30
    resp = ""
    while time.time() < deadline:
        chunk = ser.read_all().decode(errors="ignore")
        resp += chunk
        if "DONE" in resp:
            break
        time.sleep(0.1)

    print(f"发送: {cmd}")
    print(f"响应: {resp.strip()}")
    ser.close()


if __name__ == "__main__":
    main()
