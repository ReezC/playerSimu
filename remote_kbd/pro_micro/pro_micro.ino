// Pro Micro 远程键盘固件
// 通过 USB CDC 串口接收文本指令，模拟成 HID 键盘输出。
// 烧录：Tools->Board->SparkFun Pro Micro, 处理器 ATmega32U4 (3.3V, 8MHz)
//       点「上传」后快速双击 RESET（RST 短接 GND 两次）进入 bootloader。
//
// 指令（每条以换行结束）：
//   PRESS key        按下不放（移动）
//   RELEASE key      松开
//   RELEASEALL       松开所有键
//   TAP key ms       点按（按下 -> 等 ms -> 松开）
//   FIX key interval count      固定间隔连按
//   RND key min max count       随机间隔连按
//   HOLD key ms      长按
//   COMBO k1 k2 ms   两键同按
// 键名：单字符(字母/数字) 或 LEFT/RIGHT/UP/DOWN/SPACE/ENTER/ESC/TAB/CTRL/SHIFT/ALT/F1..F12 等

#include <Keyboard.h>
#include <Mouse.h>

#define MAX_HELD 6
int held[MAX_HELD];
int heldCount = 0;

void setup() {
  Serial.begin(115200);
  Keyboard.begin();
  Mouse.begin();
  randomSeed(analogRead(A0));
  Serial.println("READY");
}

// 键名 -> Keyboard 键码；未知返回 -1
int keyToCode(String k) {
  k.trim();
  if (k.length() == 1) {
    char c = k.charAt(0);
    if (c >= 32 && c <= 126) return (int)c;   // 可打印 ASCII 直接当键码
  }
  k.toUpperCase();
  if (k == "SPACE") return ' ';
  if (k == "ENTER" || k == "RETURN") return KEY_RETURN;
  if (k == "ESC") return KEY_ESC;
  if (k == "TAB") return KEY_TAB;
  if (k == "BACKSPACE") return KEY_BACKSPACE;
  if (k == "UP") return KEY_UP_ARROW;
  if (k == "DOWN") return KEY_DOWN_ARROW;
  if (k == "LEFT") return KEY_LEFT_ARROW;
  if (k == "RIGHT") return KEY_RIGHT_ARROW;
  if (k == "DEL") return KEY_DELETE;
  if (k == "INSERT") return KEY_INSERT;
  if (k == "HOME") return KEY_HOME;
  if (k == "END") return KEY_END;
  if (k == "PGUP") return KEY_PAGE_UP;
  if (k == "PGDN") return KEY_PAGE_DOWN;
  if (k == "GRAVE") return '`';   // 反引号键（` / ~）
  if (k == "CTRL") return KEY_LEFT_CTRL;
  if (k == "SHIFT") return KEY_LEFT_SHIFT;
  if (k == "ALT") return KEY_LEFT_ALT;
  if (k == "GUI" || k == "WIN") return KEY_LEFT_GUI;
  if (k == "RCTRL") return KEY_RIGHT_CTRL;
  if (k == "RSHIFT") return KEY_RIGHT_SHIFT;
  if (k == "RALT") return KEY_RIGHT_ALT;
  if (k == "F1") return KEY_F1;
  if (k == "F2") return KEY_F2;
  if (k == "F3") return KEY_F3;
  if (k == "F4") return KEY_F4;
  if (k == "F5") return KEY_F5;
  if (k == "F6") return KEY_F6;
  if (k == "F7") return KEY_F7;
  if (k == "F8") return KEY_F8;
  if (k == "F9") return KEY_F9;
  if (k == "F10") return KEY_F10;
  if (k == "F11") return KEY_F11;
  if (k == "F12") return KEY_F12;
  return -1;
}

// 按空格拆 token，返回第 i 个（0 起）
String token(String s, int i) {
  int start = 0, idx = 0;
  for (int p = 0; p <= (int)s.length(); p++) {
    if (p == s.length() || s.charAt(p) == ' ') {
      if (idx == i) return s.substring(start, p);
      idx++;
      start = p + 1;
    }
  }
  return "";
}

bool addHeld(int c) {
  for (int i = 0; i < heldCount; i++) if (held[i] == c) return false;
  if (heldCount >= MAX_HELD) return false;
  held[heldCount++] = c;
  return true;
}

bool removeHeld(int c) {
  for (int i = 0; i < heldCount; i++) if (held[i] == c) {
    held[i] = held[--heldCount];
    return true;
  }
  return false;
}

void releaseAll() {
  for (int i = 0; i < heldCount; i++) Keyboard.release(held[i]);
  heldCount = 0;
}

void loop() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;
  String head = token(cmd, 0);
  head.toUpperCase();

  if (head == "PRESS") {
    int code = keyToCode(token(cmd, 1));
    if (code >= 0 && addHeld(code)) Keyboard.press(code);
    Serial.println("DONE");
  }
  else if (head == "RELEASE") {
    int code = keyToCode(token(cmd, 1));
    if (code >= 0 && removeHeld(code)) Keyboard.release(code);
    Serial.println("DONE");
  }
  else if (head == "RELEASEALL") {
    releaseAll();
    Serial.println("DONE");
  }
  else if (head == "MOVE") {
    int dx = token(cmd, 1).toInt();
    int dy = token(cmd, 2).toInt();
    Mouse.move(dx, dy, 0);
    Serial.println("DONE");
  }
  else if (head == "CLICK") {
    String btn = token(cmd, 1);
    btn.toUpperCase();
    if (btn == "RIGHT") Mouse.click(MOUSE_RIGHT);
    else if (btn == "MIDDLE") Mouse.click(MOUSE_MIDDLE);
    else Mouse.click(MOUSE_LEFT);
    Serial.println("DONE");
  }
  else if (head == "SCROLL") {
    int n = token(cmd, 1).toInt();
    Mouse.move(0, 0, n);
    Serial.println("DONE");
  }
  else if (head == "PRESSM") {
    String btn = token(cmd, 1);
    btn.toUpperCase();
    if (btn == "RIGHT") Mouse.press(MOUSE_RIGHT);
    else if (btn == "MIDDLE") Mouse.press(MOUSE_MIDDLE);
    else Mouse.press(MOUSE_LEFT);
    Serial.println("DONE");
  }
  else if (head == "RELEASEM") {
    String btn = token(cmd, 1);
    btn.toUpperCase();
    if (btn == "RIGHT") Mouse.release(MOUSE_RIGHT);
    else if (btn == "MIDDLE") Mouse.release(MOUSE_MIDDLE);
    else Mouse.release(MOUSE_LEFT);
    Serial.println("DONE");
  }
  else if (head == "TAP") {
    int code = keyToCode(token(cmd, 1));
    int ms = token(cmd, 2).toInt();
    if (code >= 0) {
      Keyboard.press(code);
      delay(ms);
      Keyboard.release(code);
    }
    Serial.println("DONE");
  }
  else if (head == "FIX") {
    int code = keyToCode(token(cmd, 1));
    int interval = token(cmd, 2).toInt();
    int count = token(cmd, 3).toInt();
    for (int i = 0; i < count && code >= 0; i++) {
      Keyboard.press(code);
      delay(5);
      Keyboard.release(code);
      if (i < count - 1) delay(interval);
    }
    Serial.println("DONE");
  }
  else if (head == "RND") {
    int code = keyToCode(token(cmd, 1));
    int minMs = token(cmd, 2).toInt();
    int maxMs = token(cmd, 3).toInt();
    int count = token(cmd, 4).toInt();
    for (int i = 0; i < count && code >= 0; i++) {
      Keyboard.press(code);
      delay(5);
      Keyboard.release(code);
      if (i < count - 1) delay(random(minMs, maxMs));
    }
    Serial.println("DONE");
  }
  else if (head == "HOLD") {
    int code = keyToCode(token(cmd, 1));
    int ms = token(cmd, 2).toInt();
    if (code >= 0) {
      Keyboard.press(code);
      delay(ms);
      Keyboard.release(code);
    }
    Serial.println("DONE");
  }
  else if (head == "COMBO") {
    int c1 = keyToCode(token(cmd, 1));
    int c2 = keyToCode(token(cmd, 2));
    int ms = token(cmd, 3).toInt();
    if (c1 >= 0 && c2 >= 0) {
      Keyboard.press(c1);
      Keyboard.press(c2);
      delay(ms);
      Keyboard.release(c1);
      Keyboard.release(c2);
    }
    Serial.println("DONE");
  }
  else {
    Serial.println("ERR");
  }
}
